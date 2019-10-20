const electron = require('electron')
const ipcRenderer = electron.ipcRenderer;
let result = document.querySelector('#result')
const table = document.querySelector('table');

let submitWindowBtn = document.getElementById('submitAddWindow');
submitWindowBtn.addEventListener('click', function(e){
    <!--so that it doesnt save in a file-->

    e.preventDefault();
    ipcRenderer.send('submit:addWindow');

});

//catch add item
ipcRenderer.on('item:add', function(e, item){
  ul.className = 'collection';
  //apply collection class from materialize only when
  //there is an element
  const li = document.createElement('li');
  li.className = 'collection-item';
  const itemText = document.createTextNode(item);
  li.appendChild(itemText);
  ul.appendChild(li);
});
