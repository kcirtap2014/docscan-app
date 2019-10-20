const electron = require('electron');
const {ipcRenderer} = electron;

const form = document.querySelector('form');
let {PythonShell} = require('python-shell')

form.addEventListener('submit', submitForm);

function submitForm(e){
  <!--so that it doesnt save in a file-->

  e.preventDefault();
  const item = document.querySelector('#item').value;

  PythonShell.run('utils.py', null, function (err) {
    if (err) throw err;
    console.log('finished');
  });

  })


  ipcRenderer.send('item:add', item);

  <!--ipcRenderer send the object to mainWindow, mainWindow catches it-->

}
