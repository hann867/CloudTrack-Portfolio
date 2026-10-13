# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.


# CloudTrack Frontend

This directory contains the CloudTrack web interface, built with React and Vite.

The frontend provides the user interface for uploading images, viewing image-processing results, and interacting with the CloudTrack backend modules.

## Prerequisites

Before starting the frontend, make sure the following are installed on your computer:

- Node.js
- npm (included with Node.js)
- The CloudTrack backend dependencies, if you also want to use the API and image-processing features

The commands below assume the project is located at:

```text
D:\cloudchamber
```

## Setup & Startup Instructions

Open a new PowerShell terminal for the frontend.

### 1. Go to the Frontend Folder

```powershell
cd D:\cloudchamber\frontend
```

### 2. Install Frontend Dependencies

Install the packages listed in `package.json`:

```powershell
npm install
```

You normally only need to run `npm install` when setting up the project for the first time or when frontend dependencies have changed.

### 3. Start the Vite Development Server

```powershell
npm run dev
```

Vite will display the local frontend URL in the terminal. It is commonly:

```text
http://localhost:5173
```

Open the URL shown by Vite in your browser.

Keep this terminal open while using the frontend.

## Quick Start

For a new frontend setup:

```powershell
cd D:\cloudchamber\frontend
npm install
npm run dev
```

For later runs, if the dependencies are already installed:

```powershell
cd D:\cloudchamber\frontend
npm run dev
```

## Running Frontend and Backend Together

The frontend should be used together with the FastAPI backend when testing the complete CloudTrack workflow.

Use two separate PowerShell terminals.

### Terminal 1 — Backend

```powershell
cd D:\cloudchamber
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload
```

The backend will normally run at:

```text
http://127.0.0.1:8000
```

### Terminal 2 — Frontend

```powershell
cd D:\cloudchamber\frontend
npm run dev
```

Then open the frontend URL displayed by Vite in your browser.

> If this is the first time setting up the project, create the backend virtual environment and install its requirements before starting the backend. See `backendREADME.md` for the complete backend setup.

## Development Workflow

A typical development session is:

1. Start the FastAPI backend in the first terminal.
2. Start the Vite frontend in the second terminal.
3. Open the frontend in your browser.
4. Upload or process images through the CloudTrack interface.
5. Keep both terminals running while developing or testing the application.

Vite uses hot module replacement during development, so most frontend code changes will appear in the browser automatically after the file is saved.

## Stopping the Frontend

To stop the Vite development server, return to the frontend terminal and press:

```text
Ctrl + C
```
