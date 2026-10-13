import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import PageContainer from './components/layout/PageContainer';
import Dashboard from './pages/Dashboard';
import ImageProcessing from './pages/ImageProcessing';
import Detection from './pages/Detection';
import Calibration from './pages/Calibration';
import Results from './pages/Results';
import ProcessingWorkflow from './pages/ProcessingWorkflow';
import VideoProcessing from './pages/VideoProcessing';

function App() {
  return (
    <Router>
      <PageContainer>
        <Routes>
          <Route path="/" element={<Navigate to="/workflow" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/processing" element={<ImageProcessing />} />
          <Route path="/workflow" element={<ProcessingWorkflow />} />
          <Route path="/video-processing" element={<VideoProcessing />} />
          <Route path="/detection" element={<Detection />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </PageContainer>
    </Router>
  );
}

export default App;
