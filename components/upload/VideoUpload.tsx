"use client"

import { useState } from "react"
import { Upload, FileVideo, CheckCircle, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"
import { uploadVideo } from "@/lib/api"
import { cn } from "@/lib/utils"

export function VideoUpload() {
    const [file, setFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [result, setResult] = useState<string | null>(null);
    const { toast } = useToast();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setIsUploading(true);
        try {
            const response = await uploadVideo(file);
            setResult(response.text);
            toast({
                title: "Upload Complete",
                description: "Video processed successfully.",
            });
        } catch (error) {
            toast({
                variant: "destructive",
                title: "Upload Failed",
                description: "Could not process video. Please try again.",
            });
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="video">Upload Video</Label>
                <Input id="video" type="file" accept="video/*" onChange={handleFileChange} />
            </div>

            {file && (
                <Card>
                    <CardContent className="pt-6 flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="p-2 bg-primary/10 rounded-full">
                                <FileVideo className="w-6 h-6 text-primary" />
                            </div>
                            <div>
                                <p className="font-medium">{file.name}</p>
                                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            </div>
                        </div>
                        <Button onClick={handleUpload} disabled={isUploading}>
                            {isUploading ? "Processing..." : "Translate Video"}
                        </Button>
                    </CardContent>
                </Card>
            )}

            {result && (
                <Card className="bg-muted/50">
                    <CardContent className="pt-6 space-y-2">
                        <h3 className="font-semibold flex items-center gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500" />
                            Translation Result
                        </h3>
                        <p className="text-lg leading-relaxed">{result}</p>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
