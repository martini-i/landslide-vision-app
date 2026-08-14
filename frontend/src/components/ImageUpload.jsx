import { useRef, useState } from "react";

export default function ImageUpload({ onFileSelected, previewUrl }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function handleFiles(files) {
    if (files && files.length > 0) {
      onFileSelected(files[0]);
    }
  }

  return (
    <div
      className={`upload-dropzone${dragOver ? " drag-over" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="Uploaded slope" className="upload-preview" />
      ) : (
        <p>Drop a slope image here, or click to upload</p>
      )}
    </div>
  );
}
