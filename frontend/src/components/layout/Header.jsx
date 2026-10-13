import { Bell, Search, User } from 'lucide-react';
import './Header.css';

function Header() {
  return (
    <header className="header glass">
      <div className="search-bar">
        <Search size={18} className="search-icon" />
        <input type="text" placeholder="Search datasets, runs..." className="search-input" />
      </div>
      
      <div className="header-actions">
        <button className="icon-btn">
          <Bell size={20} />
          <span className="badge">2</span>
        </button>
        <div className="user-profile">
          <div className="avatar">
            <User size={18} />
          </div>
          <span>Researcher</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
