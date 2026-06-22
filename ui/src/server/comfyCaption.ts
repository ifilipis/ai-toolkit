import fs from 'fs';
import zlib from 'zlib';

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function readNullTerminated(data: Buffer, start: number): { value: Buffer; next: number } | null {
  const end = data.indexOf(0, start);
  if (end < 0) return null;
  return { value: data.subarray(start, end), next: end + 1 };
}

function decodeText(buffer: Buffer, encoding: BufferEncoding = 'utf8'): string {
  try {
    return buffer.toString(encoding);
  } catch {
    return '';
  }
}

function parseTextChunk(data: Buffer): [string, string] | null {
  const keyword = readNullTerminated(data, 0);
  if (!keyword) return null;
  return [decodeText(keyword.value, 'latin1'), decodeText(data.subarray(keyword.next))];
}

function parseCompressedTextChunk(data: Buffer): [string, string] | null {
  const keyword = readNullTerminated(data, 0);
  if (!keyword || keyword.next >= data.length) return null;
  const compressionMethod = data[keyword.next];
  if (compressionMethod !== 0) return null;

  try {
    const inflated = zlib.inflateSync(data.subarray(keyword.next + 1));
    return [decodeText(keyword.value, 'latin1'), decodeText(inflated)];
  } catch {
    return null;
  }
}

function parseInternationalTextChunk(data: Buffer): [string, string] | null {
  const keyword = readNullTerminated(data, 0);
  if (!keyword || keyword.next + 2 > data.length) return null;

  const compressionFlag = data[keyword.next];
  const compressionMethod = data[keyword.next + 1];
  if (compressionFlag !== 0 && compressionFlag !== 1) return null;
  if (compressionFlag === 1 && compressionMethod !== 0) return null;

  const languageTag = readNullTerminated(data, keyword.next + 2);
  if (!languageTag) return null;
  const translatedKeyword = readNullTerminated(data, languageTag.next);
  if (!translatedKeyword) return null;

  try {
    const textData = data.subarray(translatedKeyword.next);
    const text = compressionFlag === 1 ? zlib.inflateSync(textData) : textData;
    return [decodeText(keyword.value, 'latin1'), decodeText(text)];
  } catch {
    return null;
  }
}

function readPngTextMetadata(filePath: string): Record<string, string> {
  let png: Buffer;
  try {
    png = fs.readFileSync(filePath);
  } catch {
    return {};
  }

  if (png.length < PNG_SIGNATURE.length || !png.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    return {};
  }

  const metadata: Record<string, string> = {};
  let offset = PNG_SIGNATURE.length;

  while (offset + 12 <= png.length) {
    const length = png.readUInt32BE(offset);
    const typeStart = offset + 4;
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    const nextOffset = dataEnd + 4;

    if (dataEnd > png.length || nextOffset > png.length) break;

    const type = png.subarray(typeStart, typeStart + 4).toString('ascii');
    const data = png.subarray(dataStart, dataEnd);
    let parsed: [string, string] | null = null;

    if (type === 'tEXt') parsed = parseTextChunk(data);
    else if (type === 'zTXt') parsed = parseCompressedTextChunk(data);
    else if (type === 'iTXt') parsed = parseInternationalTextChunk(data);

    if (parsed) {
      const [key, value] = parsed;
      metadata[key] = value;
    }

    if (type === 'IEND') break;
    offset = nextOffset;
  }

  return metadata;
}

function extractComfyuiPromptFromWorkflow(value: string | undefined): string {
  if (!value) return '';

  let workflow: any;
  try {
    workflow = JSON.parse(value.trim());
  } catch {
    return '';
  }
  if (!workflow || typeof workflow !== 'object' || !Array.isArray(workflow.nodes)) return '';

  const positiveCandidates: string[] = [];
  const userPromptCandidates: string[] = [];

  for (const node of workflow.nodes) {
    if (!node || typeof node !== 'object') continue;

    const nodeType = `${node.type ?? ''}`.toLowerCase();
    const title = `${node.title ?? ''}`.toLowerCase();
    if (title.includes('negative')) continue;

    if (nodeType === 'cliptextencode' || title.includes('positive prompt')) {
      const text = Array.isArray(node.widgets_values) ? node.widgets_values[0] : undefined;
      if (typeof text === 'string' && text.trim()) positiveCandidates.push(text.trim());
    }

    if (title.includes('user prompt')) {
      const text = Array.isArray(node.widgets_values) ? node.widgets_values[0] : undefined;
      if (typeof text === 'string' && text.trim()) userPromptCandidates.push(text.trim());
    }
  }

  return positiveCandidates[0] || userPromptCandidates[0] || '';
}

export function getComfyuiCaptionFromPngMetadata(filePath: string): string {
  const metadata = readPngTextMetadata(filePath);
  return extractComfyuiPromptFromWorkflow(metadata.workflow);
}

export function getComfyuiCaptionsFromPngMetadata(filePaths: string[]): Record<string, string> {
  const captions: Record<string, string> = {};
  for (const filePath of filePaths) {
    captions[filePath] = getComfyuiCaptionFromPngMetadata(filePath);
  }
  return captions;
}
