import { useState, useEffect } from 'react';
import { Activity, Database, Image as ImageIcon } from 'lucide-react';
import { apiService } from '../services/api';

function Dashboard() {
  const [connectionStatus, setConnectionStatus] = useState('checking'); // 'checking', 'connected', 'error'
  const [errorMessage, setErrorMessage] = useState('');
  const [backendService, setBackendService] = useState('');

  useEffect(() => {
    apiService.checkHealth()
      .then(data => {
        if (data.status === 'ok') {
          setConnectionStatus('connected');
          setBackendService(data.service);
        } else {
          setConnectionStatus('error');
          setErrorMessage('Received unexpected status from server.');
        }
      })
      .catch(err => {
        setConnectionStatus('error');
        setErrorMessage(err.message);
      });
  }, []);

  return (
    <div className="page-dashboard">
      <h1 className="page-title">Dashboard</h1>
      <p className="page-description">Overview of the CloudTrack system and API status.</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <div style={{ padding: '10px', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', color: 'var(--accent-primary)' }}>
              <Activity size={24} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500' }}>Backend API</h3>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <div 
              className="status-dot" 
              style={{ 
                backgroundColor: connectionStatus === 'connected' ? '#10b981' : 
                               connectionStatus === 'error' ? '#ef4444' : '#f59e0b',
                boxShadow: `0 0 8px ${connectionStatus === 'connected' ? '#10b981' : 
                                    connectionStatus === 'error' ? '#ef4444' : '#f59e0b'}`
              }}
            ></div>
            <span style={{ fontSize: '1.25rem', fontWeight: '600' }}>
              {connectionStatus === 'connected' ? 'Backend Status: Connected' : 
               connectionStatus === 'error' ? 'Connection Failed' : 'Connecting...'}
            </span>
          </div>

          {connectionStatus === 'error' ? (
            <p style={{ color: '#ef4444', fontSize: '0.85rem' }}>
              {errorMessage}
            </p>
          ) : (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              {backendService || 'Establishing connection...'}
            </p>
          )}
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <div style={{ padding: '10px', backgroundColor: 'rgba(168, 85, 247, 0.1)', borderRadius: '8px', color: '#a855f7' }}>
              <ImageIcon size={24} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500' }}>Datasets</h3>
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: '600' }}>3</div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '8px' }}>
            Available for processing
          </p>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <div style={{ padding: '10px', backgroundColor: 'rgba(234, 179, 8, 0.1)', borderRadius: '8px', color: '#eab308' }}>
              <Database size={24} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500' }}>Processed Runs</h3>
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: '600' }}>12</div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '8px' }}>
            Awaiting analysis
          </p>
        </div>

      </div>
    </div>
  );
}

export default Dashboard;
