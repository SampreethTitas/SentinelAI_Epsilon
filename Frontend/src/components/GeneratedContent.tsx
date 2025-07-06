
import { useState } from "react";
import { Copy, Download } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface GeneratedContentProps {
  content: string;
  isGenerating: boolean;
}

const GeneratedContent = ({ content, isGenerating }: GeneratedContentProps) => {
  const { toast } = useToast();
  const [showFull, setShowFull] = useState(false);

  const copyContent = () => {
    navigator.clipboard.writeText(content);
    toast({ title: "Content copied to clipboard ✨" });
  };

  const downloadContent = () => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated-content.txt';
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Content downloaded 📄" });
  };

  const truncatedContent = content.length > 300 ? content.substring(0, 300) + "..." : content;
  const displayContent = showFull ? content : truncatedContent;

  return (
    <Card className="bg-black/40 backdrop-blur-sm border-neon-green glow-border-green animate-fade-in">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-neon-green flex items-center">
            <span className="mr-2">✨</span>
            Generated Content
          </h3>
          <div className="flex space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={copyContent}
              className="text-neon-green hover:bg-neon-green/20"
            >
              <Copy size={16} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={downloadContent}
              className="text-neon-green hover:bg-neon-green/20"
            >
              <Download size={16} />
            </Button>
          </div>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
          <pre className="text-gray-300 whitespace-pre-wrap font-mono text-sm leading-relaxed">
            {isGenerating ? (
              <span className="animate-pulse">Generating content...</span>
            ) : (
              displayContent
            )}
          </pre>
          
          {content.length > 300 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFull(!showFull)}
              className="mt-3 text-neon-teal hover:text-white"
            >
              {showFull ? "Show Less" : "Show More"}
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
};

export default GeneratedContent;
