import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Database, Play, Clock, CheckCircle, XCircle, Trash2, Copy, RefreshCw, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Separator } from "@/components/ui/separator";

// Type definitions for API responses
interface DataValutOperation {
    job_id: string;
    status: "pending" | "running" | "completed" | "failed" | "cancelled";
    message: string;
    progress?: number;
    timestamp: string;
    started_at?: string;
    completed_at?: string;
    result?: {
        database_details: {
            dbname: string;
            host: string;
            port: number;
            user: string;
            password: string;
        };
        connection_string: string;
        database_name: string;
    };
    error?: string;
}

interface DataValutResponse {
    success: boolean;
    message: string;
    job_id: string;
    status: string;
    timestamp: string;
    estimated_time?: string;
}

const DataValut = () => {
    const navigate = useNavigate();
    const { toast } = useToast();

    const [sourceConnection, setSourceConnection] = useState("");
    const [filterPrompt, setFilterPrompt] = useState("");
    const [isCreating, setIsCreating] = useState(false);
    const [currentOperation, setCurrentOperation] = useState<DataValutOperation | null>(null);
    const [websocket, setWebsocket] = useState<WebSocket | null>(null);
    const [connectionStatus, setConnectionStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
    const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

    // Cleanup WebSocket connection and polling on unmount
    useEffect(() => {
        return () => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.close();
            }
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
        };
    }, [websocket, pollingInterval]);

    // Fallback polling for job status if WebSocket fails
    const startPolling = (jobId: string) => {
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }

        const interval = setInterval(async () => {
            try {
                const response = await fetch(`http://127.0.0.1:8002/datavalut/job/${jobId}`);
                if (response.ok) {
                    const jobData = await response.json();
                    setCurrentOperation(prev => {
                        if (!prev) return null;
                        return {
                            ...prev,
                            status: jobData.status,
                            message: jobData.message,
                            progress: jobData.progress,
                            result: jobData.result,
                            error: jobData.error
                        };
                    });

                    // Stop polling if job is complete
                    if (jobData.status === "completed" || jobData.status === "failed" || jobData.status === "cancelled") {
                        clearInterval(interval);
                        setPollingInterval(null);
                    }
                }
            } catch (error) {
                console.error("Polling error:", error);
            }
        }, 2000); // Poll every 2 seconds

        setPollingInterval(interval);
    };

    const connectToJobUpdates = (jobId: string) => {
        if (websocket) {
            websocket.close(); // Close existing connection
        }

        setConnectionStatus("connecting");
        const ws = new WebSocket(`ws://127.0.0.1:8002/ws/job/${jobId}`);

        ws.onopen = () => {
            console.log(`Connected to operation updates`);
            setConnectionStatus("connected");
        };

        ws.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                console.log("WebSocket update received:", update);

                if (update.type === "job_update" || update.type === "job_status") {
                    // Update the current operation with all fields from the update
                    setCurrentOperation(prev => {
                        if (!prev) return null;

                        const newState = {
                            ...prev,
                            status: update.status || prev.status,
                            message: update.message || prev.message,
                            progress: update.progress !== undefined ? update.progress : prev.progress,
                            timestamp: update.timestamp || prev.timestamp,
                            started_at: update.started_at || prev.started_at,
                            completed_at: update.completed_at || prev.completed_at,
                            result: update.result || prev.result,
                            error: update.error || prev.error
                        };

                        console.log("Operation state updated:", newState);
                        return newState;
                    });

                    // Show toast for completion or failure
                    if (update.status === "completed") {
                        toast({
                            title: "Database Created! ✅",
                            description: `Operation completed successfully`,
                        });
                    } else if (update.status === "failed") {
                        toast({
                            title: "Operation Failed ❌",
                            description: `Operation failed: ${update.error || "Unknown error"}`,
                            variant: "destructive"
                        });
                    }
                }
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
            }
        };

        ws.onclose = () => {
            console.log(`Disconnected from operation updates`);
            setConnectionStatus("disconnected");
            setWebsocket(null);

            // Start polling as fallback if we have an active operation
            setTimeout(() => {
                setCurrentOperation(prev => {
                    if (prev && (prev.status === "pending" || prev.status === "running")) {
                        console.log("WebSocket disconnected, starting fallback polling...");
                        startPolling(prev.job_id);
                    }
                    return prev;
                });
            }, 1000);
        };

        ws.onerror = (error) => {
            console.error(`WebSocket error for operation:`, error);
            setConnectionStatus("disconnected");

            // Start polling as fallback
            setTimeout(() => {
                setCurrentOperation(prev => {
                    if (prev && (prev.status === "pending" || prev.status === "running")) {
                        console.log("WebSocket error, starting fallback polling...");
                        startPolling(prev.job_id);
                    }
                    return prev;
                });
            }, 1000);
        };

        setWebsocket(ws);
    };

    const handleCreateDatabase = async () => {
        if (!sourceConnection.trim()) {
            toast({
                title: "Error",
                description: "Please enter a source database connection string",
                variant: "destructive"
            });
            return;
        }

        if (!filterPrompt.trim()) {
            toast({
                title: "Error",
                description: "Please enter a filter prompt",
                variant: "destructive"
            });
            return;
        }

        // Check if there's already an active operation
        if (currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running")) {
            toast({
                title: "Operation in Progress",
                description: "Please wait for the current database creation to complete before starting another one.",
                variant: "destructive"
            });
            return;
        }

        setIsCreating(true);
        try {
            const response = await fetch("http://127.0.0.1:8002/datavalut/create-filtered-db", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    source_db_connection: sourceConnection,
                    filter_prompt: filterPrompt
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail?.message || "Failed to create database operation");
            }

            const data: DataValutResponse = await response.json();

            // Set the current operation
            setCurrentOperation({
                job_id: data.job_id,
                status: data.status as "pending" | "running" | "completed" | "failed" | "cancelled",
                message: data.message,
                timestamp: data.timestamp,
                progress: 0
            });

            toast({
                title: "Operation Started! 🚀",
                description: `Database creation started successfully`,
            });

            // Connect to this operation's updates
            connectToJobUpdates(data.job_id);

            // Also start polling as a backup (will be cleared if WebSocket works)
            setTimeout(() => {
                if (connectionStatus !== "connected") {
                    console.log("WebSocket connection taking too long, starting backup polling...");
                    startPolling(data.job_id);
                }
            }, 3000);

        } catch (error) {
            console.error("Error creating database:", error);
            const errorMessage = error instanceof Error ? error.message : "Failed to create database operation";
            toast({
                title: "Error",
                description: errorMessage,
                variant: "destructive"
            });
        } finally {
            setIsCreating(false);
        }
    };

    const handleCancelOperation = async (operationId: string) => {
        try {
            const response = await fetch(`http://127.0.0.1:8002/datavalut/job/${operationId}`, {
                method: "DELETE"
            });

            if (!response.ok) throw new Error("Failed to cancel operation");

            const data = await response.json();

            if (data.success) {
                toast({
                    title: "Operation Cancelled",
                    description: `Operation has been cancelled`,
                });
                // Update current operation status
                setCurrentOperation(prev => prev ? { ...prev, status: "cancelled" } : null);
            } else {
                toast({
                    title: "Cannot Cancel",
                    description: data.message,
                    variant: "destructive"
                });
            }
        } catch (error) {
            console.error("Error cancelling operation:", error);
            toast({
                title: "Error",
                description: "Failed to cancel operation",
                variant: "destructive"
            });
        }
    };

    const copyToClipboard = (text: string, label: string) => {
        navigator.clipboard.writeText(text);
        toast({
            title: "Copied!",
            description: `${label} copied to clipboard`,
        });
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case "pending": return <Clock className="w-4 h-4" />;
            case "running": return <RefreshCw className="w-4 h-4 animate-spin" />;
            case "completed": return <CheckCircle className="w-4 h-4" />;
            case "failed": return <XCircle className="w-4 h-4" />;
            case "cancelled": return <XCircle className="w-4 h-4" />;
            default: return <Clock className="w-4 h-4" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "pending": return "bg-yellow-500";
            case "running": return "bg-blue-500";
            case "completed": return "bg-green-500";
            case "failed": return "bg-red-500";
            case "cancelled": return "bg-gray-500";
            default: return "bg-gray-500";
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 cyber-grid">
            <div className="relative min-h-screen p-4">
                <div className="max-w-6xl mx-auto">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-8">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigate("/")}
                            className="flex items-center gap-2 text-white hover:text-gray-300"
                        >
                            <ArrowLeft className="w-4 h-4" />
                            Back to Dashboard
                        </Button>
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-green-500 rounded-lg">
                                <Database className="w-6 h-6 text-white" />
                            </div>
                            <h1 className="text-2xl font-bold text-white">DataValut</h1>
                        </div>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigate("/database-viewer")}
                            className="flex items-center gap-2 text-white hover:text-gray-300"
                        >
                            <Eye className="w-4 h-4" />
                            Database Viewer
                        </Button>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Create Database Section */}
                        <Card className="p-6 bg-black/40 backdrop-blur-sm border-gray-700">
                            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                                <Play className="w-5 h-5 text-green-500" />
                                Create Filtered Database
                            </h2>

                            {/* Operation Status Notice */}
                            {currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running") && (
                                <div className="mb-4 p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
                                    <p className="text-sm text-blue-300">
                                        🔄 <strong>Operation in Progress:</strong> Please wait for the current database creation to complete.
                                    </p>
                                </div>
                            )}

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-2 text-gray-300">
                                        Source Database Connection String
                                    </label>
                                    <Input
                                        type="text"
                                        placeholder="postgresql://user:password@host:port/database"
                                        value={sourceConnection}
                                        onChange={(e) => setSourceConnection(e.target.value)}
                                        disabled={currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running")}
                                        className="font-mono text-sm bg-gray-800/50 border-gray-600 text-white placeholder-gray-400 focus:border-green-500 disabled:opacity-50"
                                    />
                                    <p className="text-xs text-gray-400 mt-1">
                                        Enter the connection string for your source database
                                    </p>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium mb-2 text-gray-300">
                                        Filter Prompt
                                    </label>
                                    <Textarea
                                        placeholder="Send promotional email to customers from Canada."
                                        value={filterPrompt}
                                        onChange={(e) => setFilterPrompt(e.target.value)}
                                        rows={4}
                                        disabled={currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running")}
                                        className="bg-gray-800/50 border-gray-600 text-white placeholder-gray-400 focus:border-green-500 disabled:opacity-50"
                                    />
                                    <p className="text-xs text-gray-400 mt-1">
                                        Use natural language to describe the filtering criteria
                                    </p>
                                </div>

                                <Button
                                    onClick={handleCreateDatabase}
                                    disabled={
                                        isCreating ||
                                        !sourceConnection.trim() ||
                                        !filterPrompt.trim() ||
                                        (currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running"))
                                    }
                                    className="w-full bg-green-500 hover:bg-green-600 text-white disabled:opacity-50"
                                >
                                    {isCreating ? (
                                        <>
                                            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                            Creating Database...
                                        </>
                                    ) : (currentOperation && (currentOperation.status === "pending" || currentOperation.status === "running")) ? (
                                        <>
                                            <Clock className="w-4 h-4 mr-2" />
                                            Operation in Progress...
                                        </>
                                    ) : (
                                        <>
                                            <Database className="w-4 h-4 mr-2" />
                                            Create Filtered Database
                                        </>
                                    )}
                                </Button>
                            </div>
                        </Card>

                        {/* Current Operation Status */}
                        <Card className="p-6 bg-black/40 backdrop-blur-sm border-gray-700">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
                                    <Clock className="w-5 h-5 text-blue-500" />
                                    Operation Status
                                </h2>
                                {/* {currentOperation && (
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${connectionStatus === "connected" ? "bg-green-500" :
                                                connectionStatus === "connecting" ? "bg-yellow-500" : "bg-red-500"
                                            }`}></div>
                                        <span className="text-xs text-gray-400">
                                            {connectionStatus === "connected" ? "Live updates" :
                                                connectionStatus === "connecting" ? "Connecting..." : "Disconnected"}
                                        </span>
                                    </div>
                                )} */}
                            </div>

                            {!currentOperation ? (
                                <div className="text-center py-8 text-gray-400">
                                    <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                    <p>No active operation</p>
                                    <p className="text-sm">Create a filtered database to see progress here</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    <div className="border border-gray-600 rounded-lg p-3 bg-gray-800/30">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                                {getStatusIcon(currentOperation.status)}
                                                <Badge className={`${getStatusColor(currentOperation.status)} text-white`}>
                                                    {currentOperation.status.toUpperCase()}
                                                </Badge>
                                                <span className="text-sm text-gray-400">
                                                    Database Operation
                                                </span>
                                            </div>
                                            {(currentOperation.status === "pending" || currentOperation.status === "running") && (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => handleCancelOperation(currentOperation.job_id)}
                                                    className="border-gray-600 text-gray-300 hover:bg-red-600 hover:border-red-600 hover:text-white"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </Button>
                                            )}
                                        </div>

                                        <p className="text-sm text-gray-300 mb-2">{currentOperation.message}</p>

                                        {currentOperation.progress !== undefined && currentOperation.status === "running" && (
                                            <Progress value={currentOperation.progress * 100} className="mb-2" />
                                        )}

                                        {currentOperation.result && (
                                            <div className="bg-green-900/30 border border-green-700 rounded p-2 mt-2">
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-xs text-green-300 font-medium">
                                                        PostgreSQL Connection String:
                                                    </span>
                                                    <div className="flex items-center gap-2">
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => copyToClipboard(currentOperation.result!.connection_string, "Connection string")}
                                                            className="text-green-300 hover:text-green-200 hover:bg-green-800/30"
                                                        >
                                                            <Copy className="w-3 h-3 mr-1" />
                                                            Copy
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => navigate(`/database-viewer?connection=${encodeURIComponent(currentOperation.result!.connection_string)}`)}
                                                            className="text-green-300 hover:text-green-200 hover:bg-green-800/30 border border-green-500/50"
                                                        >
                                                            <Eye className="w-3 h-3 mr-1" />
                                                            View Database
                                                        </Button>
                                                    </div>
                                                </div>
                                                <div className="font-mono text-xs text-green-200 bg-green-900/20 p-2 rounded break-all">
                                                    {currentOperation.result.connection_string.replace(/:([^@]+)@/, ':****@')}
                                                </div>

                                                {/* Quick Access Button */}
                                                <div className="mt-3 pt-2 border-t border-green-700/50">
                                                    <Button
                                                        onClick={() => navigate(`/database-viewer?connection=${encodeURIComponent(currentOperation.result!.connection_string)}`)}
                                                        className="w-full bg-green-600/20 hover:bg-green-600/30 text-green-300 border border-green-500 hover:border-green-400 transition-all"
                                                        size="sm"
                                                    >
                                                        <Eye className="w-4 h-4 mr-2" />
                                                        Open Database Viewer
                                                    </Button>
                                                </div>
                                            </div>
                                        )}

                                        {currentOperation.error && (
                                            <div className="bg-red-900/30 border border-red-700 rounded p-2 mt-2">
                                                <span className="text-sm text-red-300">{currentOperation.error}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DataValut;
