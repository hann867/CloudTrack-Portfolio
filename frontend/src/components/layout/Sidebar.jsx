import { NavLink } from 'react-router-dom';
import { Image as ImageIcon, Video } from 'lucide-react';
import './Sidebar.css';

function Sidebar() {
  const navItems = [
    { path: '/workflow', label: 'Image Processing', icon: <ImageIcon size={20} /> },
    { path: '/video-processing', label: 'Video Processing', icon: <Video size={20} /> },
  ];

  return (
    <aside className="sidebar glass">
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-icon">C</div>
          <h2>CloudTrack</h2>
        </div>
      </div>
      
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink 
            key={item.path} 
            to={item.path} 
            className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}
          >
            {item.icon}
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      
    </aside>
  );
}

export default Sidebar;
