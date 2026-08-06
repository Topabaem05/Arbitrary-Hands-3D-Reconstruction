import { createHash } from 'node:crypto';
import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = process.cwd();
const DIST = join(ROOT, 'dist');
const ASSET_DIR = join(DIST, 'assets');
const ASSET_PATH = join(ASSET_DIR, 'hand_rigged_v3_lod0.glb');
const ASSET_URL = 'https://drive.google.com/uc?export=download&id=1k_tws71vEZw5WMBIvSYRKWovi3sF1BwV';
const EXPECTED_BYTES = 485752;
const EXPECTED_SHA256 = '4e251e8237cb4c30d1c7be22be14bd516cd2d633a63f00fd159d2323d140357e';
const STATIC_ENTRIES = ['index.html', 'styles.css', 'app.js', 'favicon.svg', 'src', 'models'];

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function validateGlb(bytes) {
  if (bytes.byteLength !== EXPECTED_BYTES) {
    throw new Error(`Rigged hand byte length mismatch: ${bytes.byteLength}`);
  }
  if (sha256(bytes) !== EXPECTED_SHA256) {
    throw new Error('Rigged hand SHA-256 mismatch.');
  }
  if (bytes.readUInt32LE(0) !== 0x46546c67) {
    throw new Error('Rigged hand GLB magic is invalid.');
  }
  if (bytes.readUInt32LE(4) !== 2) {
    throw new Error('Only GLB version 2 is supported.');
  }
  if (bytes.readUInt32LE(8) !== bytes.byteLength) {
    throw new Error('Rigged hand GLB declared length is invalid.');
  }
}

async function main() {
  await rm(DIST, { recursive: true, force: true });
  await mkdir(ASSET_DIR, { recursive: true });

  for (const entry of STATIC_ENTRIES) {
    await cp(join(ROOT, entry), join(DIST, entry), { recursive: true });
  }

  const response = await fetch(ASSET_URL, {
    redirect: 'follow',
    headers: { 'User-Agent': 'ACR-Hand-Lab-Vercel-Build/1.0' },
  });
  if (!response.ok) {
    throw new Error(`Rigged hand download failed with HTTP ${response.status}.`);
  }

  const bytes = Buffer.from(await response.arrayBuffer());
  validateGlb(bytes);
  await writeFile(ASSET_PATH, bytes);
  console.log(`Bundled ${bytes.byteLength} byte rigged hand GLB (${EXPECTED_SHA256}).`);
}

await main();
