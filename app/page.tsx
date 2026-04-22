"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CameraFeed } from "@/components/live/CameraFeed"
import { TranslationOutput } from "@/components/live/TranslationOutput"
import { VideoUpload } from "@/components/upload/VideoUpload"
import { useLiveTranslation } from "@/hooks/useTranslations"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Home() {
    const {
        currentText,
        confidence,
        isTranslating,
        setIsTranslating,
        processFrame,
        clearText
    } = useLiveTranslation();

    return (
        <div className="space-y-12">
            {/* Hero Section */}
            <section className="text-center space-y-6 py-12 md:py-20">
                <h1 className="text-4xl md:text-6xl font-bold tracking-tight bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
                    Sign to Text, Instantly.
                </h1>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                    Break down communication barriers with real-time American Sign Language translation powered by advanced AI.
                </p>
                <div className="flex justify-center gap-4">
                    <Button size="lg" className="rounded-full px-8" onClick={() => {
                        document.getElementById('translation-panel')?.scrollIntoView({ behavior: 'smooth' });
                    }}>
                        Start Translating <ArrowRight className="ml-2 w-4 h-4" />
                    </Button>
                    <Button size="lg" variant="outline" className="rounded-full px-8">
                        Learn More
                    </Button>
                </div>
            </section>

            {/* Main Translation Interface */}
            <section id="translation-panel" className="scroll-mt-24">
                <Tabs defaultValue="live" className="space-y-6">
                    <div className="flex justify-center">
                        <TabsList className="grid w-full max-w-md grid-cols-2">
                            <TabsTrigger value="live">Live Camera</TabsTrigger>
                            <TabsTrigger value="upload">Upload Video</TabsTrigger>
                        </TabsList>
                    </div>

                    <TabsContent value="live" className="space-y-6">
                        <div className="grid lg:grid-cols-2 gap-6 h-[600px]">
                            {/* Left: Camera */}
                            <div className="h-full">
                                <CameraFeed
                                    onFrame={processFrame}
                                    isTranslating={isTranslating}
                                    onToggleTranslate={() => setIsTranslating(!isTranslating)}
                                />
                            </div>

                            {/* Right: Output */}
                            <div className="h-full">
                                <TranslationOutput
                                    text={currentText}
                                    confidence={confidence}
                                    isTranslating={isTranslating}
                                    onClear={clearText}
                                />
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="upload">
                        <div className="max-w-2xl mx-auto">
                            <VideoUpload />
                        </div>
                    </TabsContent>
                </Tabs>
            </section>
        </div>
    );
}
