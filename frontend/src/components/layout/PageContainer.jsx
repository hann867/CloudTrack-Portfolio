import Sidebar from './Sidebar';
import './PageContainer.css';

// eslint-disable-next-line react/prop-types
function PageContainer({ children }) {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <main className="page-content">
          {children}
        </main>
      </div>
    </div>
  );
}

export default PageContainer;
