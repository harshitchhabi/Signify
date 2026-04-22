export interface TranslationResponse {
    text: string;
    confidence: number;
    partial: boolean;
    timestamp: string;
}

export interface VideoTranslationResponse {
    text: string;
    confidence: number;
    duration: number;
    createdAt: string;
}

export interface HistoryItem {
    id: string;
    text: string;
    createdAt: string;
    confidence: number;
    source: 'live' | 'upload';
}

export interface HistoryResponse {
    items: HistoryItem[];
    total: number;
}

const API_BASE = 'http://127.0.0.1:8000/api';

export async function sendFrame(imageBlob: Blob): Promise<TranslationResponse> {
    // In a real app, we'd send the blob. For now, we'll simulate a response if the backend isn't ready.
    // But the prompt says "Assume I have a backend... you just call them".
    // So I will implement the fetch call.

    // Convert blob to base64 if needed, or send as FormData. 
    // Prompt says "Body: single image frame (Base64 or Blob)".
    // Let's use FormData for Blob.

    const formData = new FormData();
    formData.append('frame', imageBlob);

    const res = await fetch(`${API_BASE}/translate/frame`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        throw new Error('Failed to translate frame');
    }

    return res.json();
}

export async function uploadVideo(videoFile: File): Promise<VideoTranslationResponse> {
    const formData = new FormData();
    formData.append('video', videoFile);

    const res = await fetch(`${API_BASE}/translate/video`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        throw new Error('Failed to upload video');
    }

    return res.json();
}

export async function getHistory(limit = 20, offset = 0): Promise<HistoryResponse> {
    const res = await fetch(`${API_BASE}/translations?limit=${limit}&offset=${offset}`);

    if (!res.ok) {
        throw new Error('Failed to fetch history');
    }

    return res.json();
}
