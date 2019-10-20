const electron = require('electron')
// Module to control application life.
const app = electron.app
// Module to create native browser window.
const BrowserWindow = electron.BrowserWindow
const Menu = electron.Menu;
const ipcMain = electron.ipcMain;


const path = require('path')
const url = require('url')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
let addWindow;

function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({width: 800, height: 600})

  // and load the index.html of the app.
  mainWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'template/mainWindow.html'),
    protocol: 'file:',
    slashes: true
  }))

  // Open the DevTools.
  mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
  mainWindow = null
  //Build menu from the template
  const mainMenu = Menu.buildFromTemplate(mainMenuTemplate);

  //Insert menu
  Menu.setApplicationMenu(mainMenu);
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q

    app.quit()
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow()
  }
})

// Handle create add mainWindow
function createAddWindow(){
  //create new mainWindow
  addWindow = new BrowserWindow(
    {
      width: 500,
      height: 300,
      title: 'Ajouter un document',
      webPreferences: {
        nodeIntegration: true,
      },
      parent: mainWindow
    });
  //load html into window
  addWindow.loadURL(require('url').format({
    pathname: path.join(__dirname, 'template/addWindow.html'),
    protocol:'file:',
    slashes: true,
  })); //file://dirname/mainWindow.html

  //Garbage collection handle
  addWindow.on('closed', function(){
    addWindow = null;
  });

  }


ipcMain.on('submit:addWindow', function(e){

  if (!addWindow){
    createAddWindow();
  }
});

// Catch item:add
ipcMain.on('item:add', function(e, item){
  // console.log(item);

  mainWindow.webContents.send('item:add', item);
  addWindow.close();
});

//Create menu template
const mainMenuTemplate = [
  {
    label:'File',
    submenu: [
      {
        label: 'Add Item',
        accelerator:process.platform == 'darwin' ? 'Command+A' : 'Ctrl+A',
        click(){
          createAddWindow();
        }
      },
      {
        label: 'Clear Items',
        click(){
          mainWindow.webContents.send('item:clear');
        }
      },
      {
        label: 'Quit',
        accelerator:process.platform == 'darwin' ? 'Command+Q' : 'Ctrl+Q',
        // ternary operator ?=if, :=else
        click(){
          app.quit();
        }
      }
    ]
  }
];

//file menubar handling. If mac, add empty object to menu
if(process.platform=='darwin'){
  mainMenuTemplate.unshift({ role: 'hide' }); //push or prepend
}

if(process.env.NODE_ENV != 'production'){
  mainMenuTemplate.push({
    label : 'Developer Tools',
    submenu: [
      {
        label:"Toggle DevTools",
        accelerator:process.platform == 'darwin' ? 'Command+I' : 'Ctrl+I',
        click(item, focusedWindow){
          focusedWindow.toggleDevTools();
        }
      },
      {
        role: 'reload'
      }
    ]
  });
}

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
