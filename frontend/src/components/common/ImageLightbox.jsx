import { useEffect } from 'react';
import { X } from 'lucide-react';
import './ImageLightbox.css';


function ImageLightbox({ label, url = '', onClose, children = null }) {
  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  return (
    <div className="image-lightbox" role="presentation" onMouseDown={onClose}>
      <section
        className="image-lightbox-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <strong title={label}>{label}</strong>
          <button type="button" aria-label="Close full picture" onClick={onClose} autoFocus>
            <X size={22} />
          </button>
        </header>
        <div className="image-lightbox-content">
          {children || <img src={url} alt={label} />}
        </div>
      </section>
    </div>
  );
}


export default ImageLightbox;
