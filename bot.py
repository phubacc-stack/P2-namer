import os
import io
import time
import asyncio
from pathlib import Path
from typing import Tuple, Dict, List

import aiohttp
import numpy as np
import cv2
from PIL import Image

import discord

# ----------------- CONFIG -----------------
TOKEN = "PUT_YOUR_BOT_TOKEN_HERE"

# Folder with Pokémon images. Each filename should include the pokemon name,
# e.g. "001_bulbasaur.png" or "bulbasaur.png". We'll derive a readable name.
POKEMON_IMAGE_FOLDER = Path("images/pokemon")

# Optional: known Poketwo bot IDs to help spot spawns (not required)
POKETWO_BOT_IDS = []  # e.g. [123456789012345678]

# Keywords to detect a spawn message (case-insensitive)
SPAWN_KEYWORDS = ["a wild", "has appeared", "appeared!", "appeared"]

# Per-channel cooldown (seconds) to avoid spam
CHANNEL_COOLDOWN = 1.5

# Maximum Hamming distance to consider a confident match.
# You can tune this. pHash is 64-bit; values ~0-10 are usually very close.
CONFIDENT_DISTANCE = 12

# Hash size (we use 8x8 DCT top-left for 64-bit hash)
PHASH_SIZE = 8
# ------------------------------------------

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

# In-memory caches
last_reply_time: Dict[int, float] = {}
preloaded_hashes: Dict[str, np.ndarray] = {}  # name -> hash bits (np.uint8 of length 64)
preload_names: List[str] = []


# ----------- IMAGE HASH (pHash) IMPLEMENTATION -----------
def image_to_phash(image: Image.Image, hash_size: int = PHASH_SIZE) -> np.ndarray:
    """
    Compute a pHash for a PIL.Image and return a boolean array of length hash_size*hash_size.
    Steps:
      - convert to grayscale
      - resize to (hash_size*4, hash_size*4) for better DCT
      - apply DCT and take top-left hash_size x hash_size
      - compare to median to get bits
    """
    # convert to grayscale
    img = image.convert("L")
    # resize (larger for better frequency resolution)
    highres = hash_size * 4
    img = img.resize((highres, highres), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    # DCT
    dct = cv2.dct(arr)
    # take top-left block
    tl = dct[:hash_size, :hash_size]
    median = np.median(tl)
    # boolean hash
    bits = (tl > median).flatten().astype(np.uint8)
    return bits  # shape (hash_size*hash_size,)


def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    """Return Hamming distance between two boolean/uint8 bit arrays of equal length."""
    if a.shape != b.shape:
        raise ValueError("Hashes must have the same shape")
    # XOR then sum
    return int(np.count_nonzero(a != b))


# ------------- PRELOAD DATASET -------------
def name_from_filename(fn: str) -> str:
    """Turn a filename into a user-friendly pokemon name."""
    # examples:
    # '001_bulbasaur.png', 'bulbasaur.png', '025_pikachu.png'
    base = Path(fn).stem
    # remove numeric prefix if present
    parts = base.split("_")
    if len(parts) > 1 and parts[0].isdigit():
        parts = parts[1:]
    name = "_".join(parts)
    # replace underscores with spaces, capitalize each word
    name = " ".join(w.capitalize() for w in name.split("_"))
    return name


async def preload_hashes(folder: Path):
    """Compute and store pHashes for all images in folder."""
    global preloaded_hashes, preload_names
    preloaded_hashes = {}
    preload_names = []

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Pokemon image folder not found: {folder}")

    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")

    print(f"Preloading {len(files)} pokemon images for hashing...")

    for p in files:
        try:
            with Image.open(p) as img:
                bits = image_to_phash(img)
                name = name_from_filename(p.name)
                preloaded_hashes[name] = bits
                preload_names.append(name)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    print(f"Preloaded {len(preloaded_hashes)} pokemon hashes.")


# ------------- HELPER: DOWNLOAD IMAGE -------------
async def download_image_bytes(url: str, session: aiohttp.ClientSession, timeout: int = 10) -> bytes | None:
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                print(f"Image download returned {resp.status} for {url}")
                return None
    except Exception as e:
        print(f"Failed to download image {url}: {e}")
        return None


# ------------- SPAWN DETECTION -------------
def message_looks_like_spawn(message: discord.Message) -> bool:
    # quick checks: either content contains a spawn keyword or embed title contains 'wild'/'appeared'
    content = (message.content or "").lower()
    for k in SPAWN_KEYWORDS:
        if k.lower() in content:
            return True
    for e in message.embeds:
        if e.title and any(x in e.title.lower() for x in ("wild", "appeared", "has appeared")):
            return True
    # optionally consider message.author is poketwo bot id
    if message.author and message.author.id in POKETWO_BOT_IDS:
        return True
    # else ignore
    return False


# ------------- MAIN MATCHING LOGIC -------------
def best_match_for_hash(query_bits: np.ndarray) -> Tuple[str, int]:
    """Return (best_name, distance) among preloaded hashes."""
    best_name = None
    best_distance = 999
    for name, bits in preloaded_hashes.items():
        d = hamming_distance_bits(query_bits, bits)
        if d < best_distance:
            best_distance = d
            best_name = name
    return best_name, best_distance


# ------------- DISCORD EVENT HANDLERS -------------
@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id={client.user.id})")
    print("Ready.")


@client.event
async def on_message(message: discord.Message):
    # ignore ourselves
    if message.author and message.author.id == client.user.id:
        return

    # quick spawn detection
    if not message_looks_like_spawn(message):
        return

    channel = message.channel
    now = time.time()
    last = last_reply_time.get(channel.id, 0.0)
    if now - last < CHANNEL_COOLDOWN:
        return

    # get image url from embed or attachments
    img_url = None
    # prefer embed image (Poketwo typically uses embed.image.url)
    if message.embeds:
        for e in message.embeds:
            if e.image and e.image.url:
                img_url = e.image.url
                break
            # sometimes the pokemon sprite is in the embed thumbnail
            if e.thumbnail and e.thumbnail.url:
                img_url = e.thumbnail.url
                break

    # fall back to attachments
    if not img_url and message.attachments:
        img_url = message.attachments[0].url

    if not img_url:
        # nothing to match
        return

    async with aiohttp.ClientSession() as session:
        data = await download_image_bytes(img_url, session)
        if not data:
            return
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            print("Failed to open downloaded image:", e)
            return

        # compute pHash for the spawn
        try:
            query_bits = image_to_phash(img)
        except Exception as e:
            print("Failed to compute phash:", e)
            return

        # find best match
        best_name, best_distance = best_match_for_hash(query_bits)

        # Decide whether to send based on distance. If not confident, still send best match (you can change).
        send_name = best_name
        is_confident = best_distance <= CONFIDENT_DISTANCE

        # Format reply text
        if is_confident:
            reply = f"{send_name}"
        else:
            # optionally indicate low confidence; you can remove the '?'
            reply = f"{send_name}"

        try:
            await channel.send(reply)
            last_reply_time[channel.id] = time.time()
            print(f"Replied in {channel} -> {reply} (distance {best_distance})")
        except discord.HTTPException as e:
            print("Failed to send message:", e)


# ------------- STARTUP -------------
async def main():
    # preload dataset hashes before connecting
    await preload_hashes(POKEMON_IMAGE_FOLDER)
    # run client
    await client.start(TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping bot...")
        
