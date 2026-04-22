"use client"

import { useEffect, useRef } from "react"
import { Copy, Trash2, Volume2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/use-toast"
import { cn } from "@/lib/utils"

interface TranslationOutputProps {
    text: string;
    confidence: number;
    isTranslating: boolean;
    onClear: () => void;
}

export function TranslationOutput({ text, confidence, isTranslating, onClear }: TranslationOutputProps) {
    const { toast } = useToast();
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [text]);

    const handleCopy = () => {
        if (!text) return;
        navigator.clipboard.writeText(text);
        toast({
            title: "Copied!",
            description: "Translation copied to clipboard.",
        });
    };

    return (
        <Card className="h-full flex flex-col min-h-[400px]">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xl font-bold">Translation</CardTitle>
                <div className="flex items-center gap-2">
                    {confidence > 0 && (
                        <Badge variant={confidence > 0.8 ? "default" : "secondary"}>
                            {Math.round(confidence * 100)}% Confidence
                        </Badge>
                    )}
                </div>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
                <div className="flex-1 p-4 rounded-lg bg-muted/50 border mb-4 overflow-y-auto max-h-[300px] min-h-[200px]">
                    {text ? (
                        <p className="text-lg leading-relaxed whitespace-pre-wrap">
                            {text}
                            {isTranslating && <span className="inline-block w-2 h-5 ml-1 bg-primary animate-pulse align-middle" />}
                        </p>
                    ) : (
                        <div className="h-full flex items-center justify-center text-muted-foreground italic">
                            {isTranslating ? "Listening for signs..." : "Start translation to see text here..."}
                        </div>
                    )}
                    <div ref={bottomRef} />
                </div>

                <div className="flex items-center justify-between gap-2">
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={handleCopy} disabled={!text}>
                            <Copy className="w-4 h-4 mr-2" />
                            Copy
                        </Button>
                        <Button variant="outline" size="sm" disabled={!text}>
                            <Volume2 className="w-4 h-4 mr-2" />
                            Speak
                        </Button>
                    </div>
                    <Button variant="ghost" size="sm" onClick={onClear} disabled={!text}>
                        <Trash2 className="w-4 h-4 mr-2" />
                        Clear
                    </Button>
                </div>
            </CardContent>
        </Card>
    );
}
