
import { AlertTriangle, CheckCircle, XCircle, Clock, Target, Zap, Flag, Lightbulb } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface ThreatAnalysisProps {
  analysis: {
    threat_level: string;
    confidence: number;
    attack_types?: string[];
    recommendation: string;
    metadata?: {
      pattern_score?: number;
      ml_score?: number;
      processing_time?: number;
      prompt_length?: number;
      content_length?: number;
      user_id?: string;
      client_id?: string;
      timestamp?: string;
      pii_types_found?: number;
      pii_count?: number;
    };
    content_type?: string;
    flagged_patterns?: string[];
    suggestions?: string[];
    compliance_score?: number;
    marketing_score?: number;
    is_malicious?: boolean;
  };
}

const ThreatAnalysis = ({ analysis }: ThreatAnalysisProps) => {
  const getThreatColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'safe': return 'green';
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      case 'critical': return 'red';
      default: return 'gray';
    }
  };

  const getThreatIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'safe':
      case 'low':
        return <CheckCircle className="text-neon-green" size={24} />;
      case 'medium':
        return <AlertTriangle className="text-yellow-500" size={24} />;
      case 'high':
      case 'critical':
        return <XCircle className="text-neon-red" size={24} />;
      default:
        return <AlertTriangle className="text-gray-500" size={24} />;
    }
  };

  const color = getThreatColor(analysis.threat_level);

  return (
    <div className="space-y-6">
      {/* Threat Level Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getThreatIcon(analysis.threat_level)}
          <div>
            <Badge 
              className={`
                text-lg px-4 py-2 font-bold animate-pulse
                ${color === 'green' ? 'bg-neon-green/20 text-neon-green border-neon-green' :
                  color === 'yellow' ? 'bg-yellow-500/20 text-yellow-500 border-yellow-500' :
                  color === 'red' ? 'bg-neon-red/20 text-neon-red border-neon-red' :
                  'bg-gray-500/20 text-gray-500 border-gray-500'}
              `}
              variant="outline"
            >
              {analysis.threat_level.toUpperCase()}
            </Badge>
          </div>
        </div>
        
        {analysis.content_type && (
          <Badge variant="secondary" className="bg-gray-700/50 text-gray-300">
            {analysis.content_type}
          </Badge>
        )}
      </div>

      {/* Confidence Gauge */}
      <Card className="bg-gray-800/30 border-gray-600 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-white font-semibold">Confidence Score</span>
          <span className={`font-bold ${
            color === 'green' ? 'text-neon-green' :
            color === 'yellow' ? 'text-yellow-500' :
            color === 'red' ? 'text-neon-red' : 'text-gray-500'
          }`}>
            {Math.round(analysis.confidence * 100)}%
          </span>
        </div>
        <Progress 
          value={analysis.confidence * 100} 
          className="h-3"
        />
      </Card>

      {/* Compliance and Marketing Scores */}
      {(analysis.compliance_score !== undefined || analysis.marketing_score !== undefined) && (
        <div className="grid grid-cols-2 gap-4">
          {analysis.compliance_score !== undefined && (
            <Card className="bg-gray-800/30 border-gray-600 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-semibold text-sm">Compliance</span>
                <span className={`font-bold ${
                  analysis.compliance_score >= 80 ? 'text-neon-green' :
                  analysis.compliance_score >= 50 ? 'text-yellow-500' :
                  'text-neon-red'
                }`}>
                  {analysis.compliance_score}%
                </span>
              </div>
              <Progress 
                value={analysis.compliance_score} 
                className="h-2"
              />
            </Card>
          )}
          {analysis.marketing_score !== undefined && (
            <Card className="bg-gray-800/30 border-gray-600 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-semibold text-sm">Marketing</span>
                <span className={`font-bold ${
                  analysis.marketing_score >= 80 ? 'text-neon-green' :
                  analysis.marketing_score >= 50 ? 'text-yellow-500' :
                  'text-neon-red'
                }`}>
                  {analysis.marketing_score}%
                </span>
              </div>
              <Progress 
                value={analysis.marketing_score} 
                className="h-2"
              />
            </Card>
          )}
        </div>
      )}

      {/* Attack Types */}
      {analysis.attack_types && analysis.attack_types.length > 0 && (
        <Card className="bg-gray-800/30 border-gray-600 p-4">
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Target className="mr-2 text-neon-red" size={18} />
            Detected Threats
          </h4>
          <div className="flex flex-wrap gap-2">
            {analysis.attack_types.map((type, index) => (
              <Badge 
                key={index}
                variant="outline"
                className="bg-neon-red/10 text-neon-red border-neon-red animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {type.replace(/_/g, ' ')}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Flagged Patterns */}
      {analysis.flagged_patterns && analysis.flagged_patterns.length > 0 && (
        <Card className="bg-gray-800/30 border-gray-600 p-4">
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Flag className="mr-2 text-yellow-500" size={18} />
            Flagged Patterns
          </h4>
          <div className="space-y-2">
            {analysis.flagged_patterns.map((pattern, index) => (
              <div 
                key={index}
                className="flex items-center space-x-2 p-2 bg-yellow-500/10 rounded border border-yellow-500/20"
              >
                <AlertTriangle size={14} className="text-yellow-500 flex-shrink-0" />
                <span className="text-gray-300 text-sm">{pattern}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Recommendation */}
      <Card className="bg-gray-800/30 border-gray-600 p-4">
        <h4 className="text-white font-semibold mb-3 flex items-center">
          <Zap className="mr-2 text-neon-teal" size={18} />
          Recommendation
        </h4>
        <p className="text-gray-300 leading-relaxed animate-typewriter">
          {analysis.recommendation}
        </p>
      </Card>

      {/* Suggestions */}
      {analysis.suggestions && analysis.suggestions.length > 0 && (
        <Card className="bg-gray-800/30 border-gray-600 p-4">
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Lightbulb className="mr-2 text-neon-teal" size={18} />
            Suggestions
          </h4>
          <ul className="space-y-2">
            {analysis.suggestions.map((suggestion, index) => (
              <li 
                key={index}
                className="flex items-start space-x-2 text-gray-300"
              >
                <span className="text-neon-teal mt-1">•</span>
                <span className="text-sm">{suggestion}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Metadata */}
      {analysis.metadata && (
        <Card className="bg-gray-800/30 border-gray-600 p-4">
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Clock className="mr-2 text-gray-400" size={18} />
            Analysis Details
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {analysis.metadata.pattern_score !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">Pattern Score:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.pattern_score.toFixed(3)}
                </span>
              </div>
            )}
            {analysis.metadata.ml_score !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">ML Score:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.ml_score.toFixed(3)}
                </span>
              </div>
            )}
            {analysis.metadata.processing_time !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">Processing Time:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.processing_time.toFixed(2)}ms
                </span>
              </div>
            )}
            {analysis.metadata.prompt_length !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">Prompt Length:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.prompt_length} chars
                </span>
              </div>
            )}
            {analysis.metadata.content_length !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">Content Length:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.content_length} chars
                </span>
              </div>
            )}
            {analysis.metadata.pii_types_found !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-400">PII Types:</span>
                <span className="text-white font-mono">
                  {analysis.metadata.pii_types_found}
                </span>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ThreatAnalysis;
