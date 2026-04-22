"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { usePreferences } from "@/hooks/usePreferences"
import { useTheme } from "next-themes"
import { Moon, Sun, Monitor } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function SettingsPage() {
    const { preferences, updatePreference, isLoaded } = usePreferences();
    const { theme, setTheme } = useTheme();

    if (!isLoaded) return null;

    return (
        <div className="space-y-8 max-w-2xl mx-auto">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
                <p className="text-muted-foreground">Manage your application preferences and appearance.</p>
            </div>

            <div className="space-y-6">
                {/* Appearance */}
                <Card>
                    <CardHeader>
                        <CardTitle>Appearance</CardTitle>
                        <CardDescription>Customize how the app looks and feels.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="space-y-2">
                            <Label>Theme</Label>
                            <div className="flex gap-2">
                                <Button
                                    variant={theme === 'light' ? 'default' : 'outline'}
                                    size="sm"
                                    className="flex-1"
                                    onClick={() => setTheme('light')}
                                >
                                    <Sun className="w-4 h-4 mr-2" /> Light
                                </Button>
                                <Button
                                    variant={theme === 'dark' ? 'default' : 'outline'}
                                    size="sm"
                                    className="flex-1"
                                    onClick={() => setTheme('dark')}
                                >
                                    <Moon className="w-4 h-4 mr-2" /> Dark
                                </Button>
                                <Button
                                    variant={theme === 'system' ? 'default' : 'outline'}
                                    size="sm"
                                    className="flex-1"
                                    onClick={() => setTheme('system')}
                                >
                                    <Monitor className="w-4 h-4 mr-2" /> System
                                </Button>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label>Text Size</Label>
                            <div className="flex gap-2">
                                {(['small', 'medium', 'large'] as const).map((size) => (
                                    <Button
                                        key={size}
                                        variant={preferences.textSize === size ? 'default' : 'outline'}
                                        size="sm"
                                        className="flex-1 capitalize"
                                        onClick={() => updatePreference('textSize', size)}
                                    >
                                        {size}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Translation Preferences */}
                <Card>
                    <CardHeader>
                        <CardTitle>Translation</CardTitle>
                        <CardDescription>Configure how translations are displayed and saved.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="flex items-center justify-between space-x-2">
                            <div className="space-y-0.5">
                                <Label htmlFor="confidence">Show Confidence Score</Label>
                                <p className="text-sm text-muted-foreground">Display the AI confidence percentage next to translations.</p>
                            </div>
                            <Switch
                                id="confidence"
                                checked={preferences.showConfidence}
                                onCheckedChange={(checked) => updatePreference('showConfidence', checked)}
                            />
                        </div>
                        <div className="flex items-center justify-between space-x-2">
                            <div className="space-y-0.5">
                                <Label htmlFor="autosave">Auto-save History</Label>
                                <p className="text-sm text-muted-foreground">Automatically save all live translations to your history.</p>
                            </div>
                            <Switch
                                id="autosave"
                                checked={preferences.autoSave}
                                onCheckedChange={(checked) => updatePreference('autoSave', checked)}
                            />
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
