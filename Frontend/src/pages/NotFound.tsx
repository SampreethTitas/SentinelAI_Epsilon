
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { AlertTriangle, Home } from "lucide-react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 cyber-grid flex items-center justify-center">
      <div className="text-center animate-fade-in">
        <div className="mb-8">
          <AlertTriangle size={120} className="mx-auto text-neon-red animate-glow-pulse mb-6" />
        </div>
        
        <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-neon-red via-neon-magenta to-neon-red bg-clip-text text-transparent">
          404
        </h1>
        <h2 className="text-2xl font-bold text-white mb-4">System Access Denied</h2>
        <p className="text-lg text-gray-300 mb-8 max-w-md mx-auto">
          The requested module or endpoint could not be found in the Sentinel.AI system.
        </p>
        
        <Button
          onClick={() => navigate('/')}
          className="bg-neon-teal/20 border border-neon-teal text-neon-teal hover:bg-neon-teal hover:text-black transition-all glow-border-teal"
        >
          <Home size={20} className="mr-2" />
          Return to Dashboard
        </Button>
        
        <div className="mt-8 text-sm text-gray-500 font-mono">
          Route: {location.pathname}
        </div>
      </div>
    </div>
  );
};

export default NotFound;
