import { useState, useEffect, useCallback, useRef } from 'react';

interface UseCameraProps {
    onFrame?: (blob: Blob) => void;
    frameRate?: number; // frames per second
}

export function useCamera({ onFrame, frameRate = 2 }: UseCameraProps = {}) {
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    const startCamera = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' },
                audio: false,
            });
            setStream(mediaStream);
            setIsStreaming(true);
        } catch (err) {
            setError(err instanceof Error ? err : new Error('Failed to access camera'));
            setIsStreaming(false);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
            setIsStreaming(false);
        }
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, [stream]);

    // Capture frames
    useEffect(() => {
        if (!isStreaming || !onFrame || !videoRef.current) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        intervalRef.current = setInterval(() => {
            if (videoRef.current && ctx) {
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
                ctx.drawImage(videoRef.current, 0, 0);

                canvas.toBlob((blob) => {
                    if (blob) onFrame(blob);
                }, 'image/jpeg', 0.8);
            }
        }, 1000 / frameRate);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [isStreaming, onFrame, frameRate]);

    return {
        videoRef,
        stream,
        error,
        isLoading,
        isStreaming,
        startCamera,
        stopCamera,
    };
}
