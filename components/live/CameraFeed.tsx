"use client"

import { useEffect, useRef } from "react"
import { Camera, CameraOff, RefreshCw, Video } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useCamera } from "@/hooks/useCamera"
import { cn } from "@/lib/utils"

interface CameraFeedProps {
    onFrame: (blob: Blob) => void;
    isTranslating: boolean;
    onToggleTranslate: () => void;
}

export function CameraFeed({ onFrame, isTranslating, onToggleTranslate }: CameraFeedProps) {
    const { videoRef, stream, error, isLoading, isStreaming, startCamera, stopCamera } = useCamera({
        onFrame,
        frameRate: 15 // Send a frame every ~66ms
    });

    // Auto-start camera on mount if desired, or let user click
    // For privacy, let's make it manual or at least show state.

    useEffect(() => {
        if (videoRef.current && stream) {
            videoRef.current.srcObject = stream;
        }
    }, [stream]);

    return (
        <div className="relative w-full h-full min-h-[400px] bg-black/5 rounded-xl overflow-hidden border">
            {/* Video Element */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={cn(
                    "w-full h-full object-cover transform -scale-x-100", // Mirror effect
                    !isStreaming && "hidden"
                )}
            />

            {/* States */}
            {!isStreaming && !isLoading && !error && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center space-y-4 bg-card/50 backdrop-blur-sm">
                    <div className="p-4 rounded-full bg-primary/10">
                        <Camera className="w-12 h-12 text-primary" />
                    </div>
                    <div className="space-y-2">
                        <h3 className="text-xl font-semibold">Ready to Translate</h3>
                        <p className="text-muted-foreground max-w-xs mx-auto">
                            Enable your camera to start translating ASL signs into text in real-time.
                        </p>
                    </div>
                    <Button size="lg" onClick={startCamera}>
                        Enable Camera
                    </Button>
                </div>
            )}

            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/80">
                    <Skeleton className="w-full h-full" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
                    </div>
                </div>
            )}

            {error && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center space-y-4 bg-destructive/5">
                    <CameraOff className="w-12 h-12 text-destructive" />
                    <div className="space-y-2">
                        <h3 className="text-xl font-semibold text-destructive">Camera Error</h3>
                        <p className="text-muted-foreground max-w-xs mx-auto">
                            {error.message || "Please allow camera access to use this feature."}
                        </p>
                    </div>
                    <Button variant="outline" onClick={startCamera}>
                        Try Again
                    </Button>
                </div>
            )}

            {/* Overlay Controls */}
            {isStreaming && (
                <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4 px-4">
                    <Button
                        variant={isTranslating ? "destructive" : "default"}
                        size="lg"
                        className="shadow-lg"
                        onClick={onToggleTranslate}
                    >
                        {isTranslating ? "Stop Translating" : "Start Translating"}
                    </Button>
                    <Button
                        variant="secondary"
                        size="icon"
                        className="shadow-lg"
                        onClick={stopCamera}
                        title="Turn off camera"
                    >
                        <CameraOff className="w-5 h-5" />
                    </Button>
                </div>
            )}

            {/* Status Indicator */}
            {isStreaming && (
                <div className="absolute top-4 right-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/50 backdrop-blur-md text-white text-xs font-medium">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        Live
                    </div>
                </div>
            )}
        </div>
    );
}
