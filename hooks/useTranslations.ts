import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { sendFrame, uploadVideo, getHistory, TranslationResponse } from '@/lib/api';
import { useState } from 'react';

export function useLiveTranslation() {
    const [currentText, setCurrentText] = useState('');
    const [confidence, setConfidence] = useState(0);
    const [isTranslating, setIsTranslating] = useState(false);

    const mutation = useMutation({
        mutationFn: sendFrame,
        onSuccess: (data) => {
            if (data.text) {
                // Append text or replace based on 'partial' flag logic
                // For this demo, we'll just append if it's new
                setCurrentText(prev => data.partial ? prev : prev + ' ' + data.text);
                setConfidence(data.confidence);
            }
        },
    });

    const processFrame = (blob: Blob) => {
        if (isTranslating && !mutation.isPending) {
            mutation.mutate(blob);
        }
    };

    const clearText = () => {
        setCurrentText('');
        setConfidence(0);
    };

    return {
        currentText,
        confidence,
        isTranslating,
        setIsTranslating,
        processFrame,
        clearText,
        isProcessing: mutation.isPending,
    };
}

export function useHistory() {
    return useQuery({
        queryKey: ['history'],
        queryFn: () => getHistory(),
    });
}
