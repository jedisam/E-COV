const { app, BrowserWindow } = require('electron');
const child = require('child_process').execFile;
const path = require('path');

function createWindows() {
  let mainWindow = new BrowserWindow({
    width: 1200,
    height: 900,
    transparent: true,
    frame: false,
    // alwaysOnTop: true,
    fullscreen: true,
    center: true,
    show: false,
    maximizable: true,
    minimizable: true,
    closable: true,
    minWidth: 1000,
    minHeight: 900,
    icon: __dirname + '/assets/icons/corona.ico',
  });
  mainWindow.loadURL('http://0.0.0.0:33507/');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });
}

app.on('ready', createWindows);
