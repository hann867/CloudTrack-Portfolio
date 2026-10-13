import { BarChart3, Clock, AlertCircle } from 'lucide-react';

function Results() {
  return (
    <div className="page-results">
      <h1 className="page-title">Results & Analytics</h1>
      <p className="page-description">View processed outputs and track classifications.</p>

      <div className="card glass" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px', color: '#ef4444' }}>
          <AlertCircle size={20} />
          <span>Analytics module is currently in development. Data shown below is mocked.</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--border-color)' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart3 size={18} color="var(--accent-primary)" />
              Recent Processing Runs
            </h3>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={14} /> Last updated: Just now
            </span>
          </div>
          
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '12px 8px', fontWeight: '500' }}>Run ID</th>
                <th style={{ padding: '12px 8px', fontWeight: '500' }}>Images Processed</th>
                <th style={{ padding: '12px 8px', fontWeight: '500' }}>Electrons Found</th>
                <th style={{ padding: '12px 8px', fontWeight: '500' }}>Alphas Found</th>
                <th style={{ padding: '12px 8px', fontWeight: '500' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan="5" style={{ padding: '32px 8px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  No runs available yet.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Results;
