import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2 } from "lucide-react"

export default function AboutPage() {
    return (
        <div className="space-y-8 max-w-3xl mx-auto">
            <div className="text-center space-y-4">
                <h1 className="text-4xl font-bold tracking-tight">About Signify</h1>
                <p className="text-xl text-muted-foreground">
                    Bridging the gap between American Sign Language and written English using advanced computer vision.
                </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle>How it Works</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4 text-muted-foreground">
                        <p>
                            Signify uses your device's camera to capture video frames in real-time. These frames are securely sent to our AI backend, which analyzes hand movements and gestures to identify ASL signs.
                        </p>
                        <p>
                            The identified signs are then converted into English text and displayed instantly on your screen, enabling seamless communication.
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Privacy & Security</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4 text-muted-foreground">
                        <p>
                            We take your privacy seriously. Video data is processed in real-time and is <strong>never stored</strong> on our servers.
                        </p>
                        <p>
                            Only the resulting text translation is saved to your history if you choose to enable that feature. You have full control over your data.
                        </p>
                    </CardContent>
                </Card>
            </div>

            <div className="space-y-4">
                <h2 className="text-2xl font-bold">Key Features</h2>
                <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
                    {[
                        "Real-time Translation",
                        "Video Upload Support",
                        "History Tracking",
                        "Dark Mode Support",
                        "Privacy Focused",
                        "Responsive Design"
                    ].map((feature) => (
                        <div key={feature} className="flex items-center gap-2 p-4 rounded-lg bg-muted/50">
                            <CheckCircle2 className="w-5 h-5 text-primary" />
                            <span className="font-medium">{feature}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
