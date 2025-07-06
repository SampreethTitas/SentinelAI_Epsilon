
const LoadingShimmer = () => {
  return (
    <div className="space-y-4 animate-pulse">
      {/* Threat Level Placeholder */}
      <div className="flex items-center space-x-3">
        <div className="w-6 h-6 bg-gray-700 rounded-full"></div>
        <div className="w-20 h-8 bg-gray-700 rounded"></div>
      </div>
      
      {/* Confidence Bar Placeholder */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <div className="w-32 h-4 bg-gray-700 rounded"></div>
          <div className="w-12 h-4 bg-gray-700 rounded"></div>
        </div>
        <div className="w-full h-3 bg-gray-700 rounded"></div>
      </div>
      
      {/* Tags Placeholder */}
      <div className="flex space-x-2">
        <div className="w-16 h-6 bg-gray-700 rounded"></div>
        <div className="w-20 h-6 bg-gray-700 rounded"></div>
        <div className="w-14 h-6 bg-gray-700 rounded"></div>
      </div>
      
      {/* Recommendation Placeholder */}
      <div className="space-y-2">
        <div className="w-28 h-4 bg-gray-700 rounded"></div>
        <div className="w-full h-4 bg-gray-700 rounded"></div>
        <div className="w-3/4 h-4 bg-gray-700 rounded"></div>
        <div className="w-1/2 h-4 bg-gray-700 rounded"></div>
      </div>
      
      {/* Animated shimmer effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-shimmer"></div>
    </div>
  );
};

export default LoadingShimmer;
