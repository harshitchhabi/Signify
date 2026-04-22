"use client"

// import { format } from "date-fns" // Removed to avoid dependency
import { Calendar, Clock, FileVideo, Video } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useHistory } from "@/hooks/useTranslations"
import { Skeleton } from "@/components/ui/skeleton"

export function HistoryList() {
    const { data, isLoading, error } = useHistory();

    if (isLoading) {
        return (
            <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                ))}
            </div>
        );
    }

    if (error) {
        return (
            <div className="text-center py-12 text-destructive">
                Failed to load history. Please try again later.
            </div>
        );
    }

    if (!data?.items?.length) {
        return (
            <div className="text-center py-12 text-muted-foreground">
                No translation history found. Start translating!
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {data.items.map((item) => (
                <Card key={item.id} className="hover:bg-muted/50 transition-colors cursor-pointer">
                    <CardHeader className="pb-2">
                        <div className="flex justify-between items-start">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                {item.source === 'live' ? <Video className="w-4 h-4" /> : <FileVideo className="w-4 h-4" />}
                                <span>{new Date(item.createdAt).toLocaleDateString()}</span>
                                <span>•</span>
                                <span>{new Date(item.createdAt).toLocaleTimeString()}</span>
                            </div>
                            <Badge variant={item.confidence > 0.8 ? "default" : "secondary"}>
                                {Math.round(item.confidence * 100)}%
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="line-clamp-2 font-medium">{item.text}</p>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}
