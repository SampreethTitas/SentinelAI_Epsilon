import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Shield, Send, Copy, Share } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import LoadingShimmer from "@/components/LoadingShimmer";
import ThreatAnalysis from "@/components/ThreatAnalysis";
import GeneratedContent from "@/components/GeneratedContent";

// Type definitions for API responses
interface GuardPromptAnalysisResult {
  is_malicious: boolean;
  threat_level: string;
  confidence: number;
  attack_types: string[];
  flagged_patterns: string[];
  processing_time: number;
  recommendation: string;
  pii_detected: Record<string, any>;
  metadata: {
    pattern_score: number;
    ml_score: number;
    prompt_length: number;
    user_id: string;
    pii_types_found: number;
    pii_count: number;
  };
}

interface GuardPromptApiResponse {
  success: boolean;
  timestamp: string;
  analysis: GuardPromptAnalysisResult;
  processing_time: number;
  request_id: string;
}

interface GenerationResult {
  generated_text: string;
}

// Replace mockAnalyzePrompt with real API call
const analyzePrompt = async (prompt: string): Promise<GuardPromptApiResponse> => {
  const response = await fetch("http://127.0.0.1:8002/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  });
  if (!response.ok) throw new Error("API error");
  return response.json();
};

// Replace mockGenerateContent with real API call
const generateContent = async (prompt: string): Promise<GenerationResult> => {
  const response = await fetch("http://127.0.0.1:8001/agents/generate/blog", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  });
  if (!response.ok) throw new Error("API error");
  return response.json();
};

const GuardPrompt = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [prompt, setPrompt] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [analysis, setAnalysis] = useState<GuardPromptAnalysisResult | null>(null);
  const [generatedContent, setGeneratedContent] = useState("");

  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Error",
        description: "Please enter a prompt to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await analyzePrompt(prompt); // Use real API
      const analysisData = response.analysis;
      setAnalysis(analysisData);
      
      const isSafe = analysisData.threat_level === 'safe';
      toast({
        title: isSafe ? "Prompt is SAFE ✅" : "Blocked 🚫",
        description: isSafe ? "This prompt appears to be safe for use" : "This prompt contains potential risks",
        variant: isSafe ? "default" : "destructive"
      });
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze prompt. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerate = async () => {
    if (!analysis || analysis.threat_level !== 'safe') return;

    setIsGenerating(true);
    try {
      const data = await generateContent(prompt); // Use real API
      setGeneratedContent(data.generated_text);
      
      toast({
        title: "Content Generated ✨",
        description: "Your safe prompt has been processed successfully"
      });
    } catch (error) {
      console.error('Generation error:', error);
      toast({
        title: "Generation Failed",
        description: "Unable to generate content. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  const copyAnalysis = () => {
    if (analysis) {
      navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
      toast({ title: "Analysis copied to clipboard" });
    }
  };

  const isSafe = analysis?.threat_level === 'safe';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 cyber-grid">
      <div className="relative min-h-screen p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/')}
              className="text-neon-teal hover:text-white hover:bg-neon-teal/20"
            >
              <ArrowLeft size={20} className="mr-2" />
              Back to Dashboard
            </Button>
          </div>
          <div className="flex items-center space-x-2">
            <Shield className="text-neon-teal" size={32} />
            <h1 className="text-3xl font-bold text-white">GuardPrompt</h1>
          </div>
        </div>

        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <Card className="bg-black/40 backdrop-blur-sm border-gray-700">
              <div className="p-6">
                <h2 className="text-xl font-bold text-white mb-4">Prompt Analysis</h2>
                
                <div className="space-y-4">
                  <Textarea
                    placeholder="Enter your marketing prompt here..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={handleKeyPress}
                    className="min-h-[200px] bg-gray-900/50 border-gray-600 text-white placeholder-gray-400 focus:border-neon-teal focus:ring-neon-teal resize-none"
                  />
                  
                  <div className="flex space-x-3">
                    <Button
                      onClick={handleAnalyze}
                      disabled={isAnalyzing || !prompt.trim()}
                      className="flex-1 bg-neon-teal/20 border border-neon-teal text-neon-teal hover:bg-neon-teal hover:text-black transition-all"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-neon-teal mr-2"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Shield size={16} className="mr-2" />
                          Check Prompt
                        </>
                      )}
                    </Button>
                    
                    <Button
                      onClick={handleGenerate}
                      disabled={!isSafe || isGenerating}
                      className={`flex-1 transition-all ${
                        isSafe 
                          ? 'bg-neon-green/20 border border-neon-green text-neon-green hover:bg-neon-green hover:text-black' 
                          : 'bg-gray-700/50 border border-gray-600 text-gray-500 cursor-not-allowed'
                      }`}
                    >
                      {isGenerating ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-neon-green mr-2"></div>
                          Generating...
                        </>
                      ) : (
                        <>
                          <Send size={16} className="mr-2" />
                          Send
                        </>
                      )}
                    </Button>
                  </div>
                  
                  <p className="text-xs text-gray-400 text-center">
                    Press Ctrl+Enter to analyze • Send is enabled only for safe prompts
                  </p>
                </div>
              </div>
            </Card>

            {/* Analysis Results */}
            <Card className="bg-black/40 backdrop-blur-sm border-gray-700">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-white">Analysis Results</h2>
                  {analysis && (
                    <div className="flex space-x-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={copyAnalysis}
                        className="text-gray-400 hover:text-neon-teal"
                      >
                        <Copy size={16} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-neon-teal"
                      >
                        <Share size={16} />
                      </Button>
                    </div>
                  )}
                </div>
                
                {isAnalyzing ? (
                  <LoadingShimmer />
                ) : analysis ? (
                  <ThreatAnalysis analysis={analysis} />
                ) : (
                  <div className="flex items-center justify-center h-64 text-gray-500 text-center">
                    <div>
                      <Shield size={48} className="mx-auto mb-4 opacity-50" />
                      <p>Enter a prompt and click "Check Prompt" to begin analysis</p>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Generated Content */}
          {generatedContent && (
            <div className="mt-8">
              <GeneratedContent content={generatedContent} isGenerating={isGenerating} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GuardPrompt;
