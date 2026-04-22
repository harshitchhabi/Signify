import { useState, useEffect } from 'react';

interface Preferences {
    textSize: 'small' | 'medium' | 'large';
    showConfidence: boolean;
    autoSave: boolean;
}

const DEFAULT_PREFERENCES: Preferences = {
    textSize: 'medium',
    showConfidence: true,
    autoSave: false,
};

export function usePreferences() {
    const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem('signify-preferences');
        if (stored) {
            try {
                setPreferences({ ...DEFAULT_PREFERENCES, ...JSON.parse(stored) });
            } catch (e) {
                console.error('Failed to parse preferences', e);
            }
        }
        setIsLoaded(true);
    }, []);

    const updatePreference = <K extends keyof Preferences>(key: K, value: Preferences[K]) => {
        const newPreferences = { ...preferences, [key]: value };
        setPreferences(newPreferences);
        localStorage.setItem('signify-preferences', JSON.stringify(newPreferences));

        // Apply side effects (like text size)
        if (key === 'textSize') {
            document.documentElement.setAttribute('data-text-size', value as string);
        }
    };

    return {
        preferences,
        updatePreference,
        isLoaded,
    };
}
