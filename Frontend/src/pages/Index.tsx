import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Shield, Search, Database, Eye, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const Index = () => {
  const navigate = useNavigate();
  const [hoveredModule, setHoveredModule] = useState<string | null>(null);

  const modules = [
    {
      id: "guardprompt",
      name: "GuardPrompt",
      description: "AI prompt risk analysis and threat detection",
      icon: Shield,
      status: "active",
      route: "/guardprompt",
      color: "teal"
    },
    {
      id: "adshield",
      name: "AdShield", 
      description: "Marketing content analysis and compliance check",
      icon: Search,
      status: "active",
      route: "/adshield",
      color: "magenta"
    },
    {
      id: "datavault",
      name: "DataVault",
      description: "Secure data storage and privacy management",
      icon: Database,
      status: "coming-soon",
      route: "#",
      color: "green"
    },
    {
      id: "trustlens",
      name: "TrustLens",
      description: "Real-time trust scoring and monitoring",
      icon: Eye,
      status: "coming-soon", 
      route: "#",
      color: "blue"
    }
  ];

  const handleModuleClick = (module: any) => {
    if (module.status === "active") {
      navigate(module.route);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 cyber-grid">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-neon-teal/5 via-transparent to-neon-magenta/5"></div>
      
      <div className="relative min-h-screen flex flex-col items-center justify-center p-8">
        {/* Header Section */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-7xl font-bold mb-6 bg-gradient-to-r from-neon-teal via-neon-magenta to-neon-teal bg-clip-text text-transparent animate-glow-pulse">
            Sentinel.AI
          </h1>
          <p className="text-2xl text-neon-teal mb-4 font-mono tracking-wider">
            AI-Driven Trust for Modern Marketing
          </p>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Sentinel.AI ensures trustworthy, regulation-safe, and bias-free marketing using AI automation.
          </p>
        </div>

        {/* Module Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl w-full">
          {modules.map((module, index) => {
            const Icon = module.icon;
            const isActive = module.status === "active";
            const isHovered = hoveredModule === module.id;

            return (
              <Card
                key={module.id}
                className={`
                  relative overflow-hidden transition-all duration-500 cursor-pointer
                  ${isActive ? 'hover:scale-105' : 'opacity-60 cursor-not-allowed'}
                  ${isHovered && isActive ? `glow-border-${module.color}` : 'border-gray-700'}
                  bg-black/40 backdrop-blur-sm
                `}
                style={{
                  animationDelay: `${index * 200}ms`
                }}
                onMouseEnter={() => setHoveredModule(module.id)}
                onMouseLeave={() => setHoveredModule(null)}
                onClick={() => handleModuleClick(module)}
              >
                {/* Animated Background */}
                <div className={`
                  absolute inset-0 opacity-0 transition-opacity duration-500
                  ${isHovered && isActive ? 'opacity-10' : ''}
                  bg-gradient-to-br from-${module.color === 'teal' ? 'neon-teal' : module.color === 'magenta' ? 'neon-magenta' : module.color === 'green' ? 'neon-green' : 'neon-blue'} to-transparent
                `}></div>

                <div className="relative p-8">
                  {/* Status Badge */}
                  <div className="absolute top-4 right-4">
                    {isActive ? (
                      <span className="px-3 py-1 text-xs font-bold bg-neon-green/20 text-neon-green border border-neon-green rounded-full">
                        ✅ ACTIVE
                      </span>
                    ) : (
                      <span className="px-3 py-1 text-xs font-bold bg-yellow-500/20 text-yellow-500 border border-yellow-500 rounded-full">
                        🚧 COMING SOON
                      </span>
                    )}
                  </div>

                  {/* Icon */}
                  <div className={`
                    w-16 h-16 mb-6 flex items-center justify-center rounded-lg
                    ${isActive ? 'bg-' + (module.color === 'teal' ? 'neon-teal' : module.color === 'magenta' ? 'neon-magenta' : module.color === 'green' ? 'neon-green' : 'neon-blue') + '/20' : 'bg-gray-700/20'}
                    ${isHovered && isActive ? 'animate-glow-pulse' : ''}
                  `}>
                    <Icon 
                      size={32} 
                      className={
                        isActive 
                          ? module.color === 'teal' ? 'text-neon-teal' 
                            : module.color === 'magenta' ? 'text-neon-magenta'
                            : module.color === 'green' ? 'text-neon-green'
                            : 'text-neon-blue'
                          : 'text-gray-500'
                      } 
                    />
                  </div>

                  {/* Content */}
                  <h3 className="text-2xl font-bold mb-3 text-white">
                    {module.name}
                  </h3>
                  <p className="text-gray-300 mb-6 leading-relaxed">
                    {module.description}
                  </p>

                  {/* Action */}
                  {isActive && (
                    <div className="flex items-center text-neon-teal group-hover:text-white transition-colors">
                      <span className="mr-2 font-semibold">Launch Module</span>
                      <ChevronRight size={20} className={`transition-transform ${isHovered ? 'translate-x-1' : ''}`} />
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-gray-500 text-sm">
          <p>2025 Sentinel.AI - Securing the Future of Marketing</p>
          <p>Made by Team XYZ of RVCE </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
