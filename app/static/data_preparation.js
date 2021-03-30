let params_list = [ 'ncols', 'nrows', 'tgt_ratio', 'train_ratio', 
                    'ntrees', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'
                ];
let model;

function updateParams(el) {
    // calculates relevant info about the dataset and displays it 
    // takes one of the input ranges and updates divs, badges and tables
    params[el.id] = el.value;
    if (el.id == 'ncols') {
        // setting number of columns/variables/predictors
        params['max_features'] = Math.round(Math.sqrt(el.value));
    }
    // update all info
    updateInfo();
    updateTable();
    // update dependent inputs
    if (el.id == 'ncols'){
        max_feats = document.getElementById('max_features');
        max_feats.setAttribute('max', params['ncols']);
        max_feats.setAttribute('value', params['max_features'] );
    } 
}

function updateInfo(){
    // updates info box and badges with dataset information
    
    // update input ranges
    for (id of params_list){
        input = document.getElementById(id);
        input.setAttribute('value', params[id]);
    }
    // update dataset badges
    let badge;
    let badge_txt;
    badges_list = ['features_badge', 'rows_badge', 'target_badge', 'split_badge'];
    for (id of badges_list){
        badge = document.getElementById(id);
        switch (id){
            case 'features_badge':
                badge_txt = document.createTextNode(params['ncols']);
                break;
            case 'rows_badge':
                badge_txt = document.createTextNode(params['nrows']);
                break;
            case 'target_badge':
                badge_txt = document.createTextNode(`${params['tgt_ratio']}%`);
                break;
            case 'split_badge':
                badge_txt = document.createTextNode(`${params['train_ratio']}-${100-params['train_ratio']}`);
                break;
        }
        while (badge.firstChild) {
            badge.removeChild(badge.firstChild);
        }
        badge.appendChild(badge_txt);
    }
    // update model param's input badges
    updateModelBadges()
    // update info text
    let info_div;
    let info_txt;
    let div_list = ['columns_info', 'rows_info', 'target_info', 'split_info']
    for (id of div_list){
        info_div = document.getElementById(id);
        switch (id){
            case 'columns_info':
                info_txt = document.createTextNode(`There are ${params['ncols']} predictive features.`);
                break;
            case 'rows_info':
                info_txt = document.createTextNode(`There are ${params['nrows']} rows in total.`);
                break;
            case 'target_info':
                info_txt = document.createTextNode(`Target share is ${params['tgt_ratio']}%.`);
                break;
            case 'split_info':
                info_txt = document.createTextNode(`Train-test split is  ${params['train_ratio']}-${100-params['train_ratio']}`);
                break;
        }
        while (info_div.firstChild) {
            info_div.removeChild(info_div.firstChild);
        }
        info_div.appendChild(info_txt);
    }
}

function updateTable() {
    // updates table with dataset information
    let train_rows = params['nrows'] * params['train_ratio'] / 100;
    let test_rows = params['nrows'] * (1 - params['train_ratio'] / 100);
    let train_tgt = train_rows * params['tgt_ratio']/100;
    let train_non_tgt = train_rows * (1 - params['tgt_ratio']/100);
    let test_tgt = test_rows * params['tgt_ratio']/100;
    let test_non_tgt = test_rows * (1 - params['tgt_ratio']/100);

    // prepare new table body
    data_table = document.getElementById('data_table');
    tbl_body = data_table.getElementsByTagName('tbody')[0];
    tbl_body?.parentElement.removeChild(tbl_body);
    tbl_body = document.createElement('tbody');
    // number of target rows
    row = document.createElement('tr');
    for (el of ['Target rows', 
                (train_tgt).toFixed(0), 
                (test_tgt).toFixed(0),
                (train_tgt+test_tgt).toFixed(0)]){   
        td = document.createElement('td');
        td.appendChild(document.createTextNode(el));
        row.appendChild(td);
    }
    tbl_body.appendChild(row);

    row = document.createElement('tr');
    for (el of ['Non-target rows', 
                (train_non_tgt).toFixed(0), 
                (test_non_tgt).toFixed(0),
                (train_non_tgt+test_non_tgt).toFixed(0)]){
        td = document.createElement('td');
        td.appendChild(document.createTextNode(el));
        row.appendChild(td);
    }
    tbl_body.appendChild(row);

    row = document.createElement('tr');
    for (el of ['Total rows', 
                (train_tgt + train_non_tgt).toFixed(0), 
                (test_tgt + test_non_tgt).toFixed(0),
                params['nrows']]) {
        td = document.createElement('td');
        td.appendChild(document.createTextNode(el));
        row.appendChild(td);
    }
    tbl_body.appendChild(row);

    data_table.appendChild(tbl_body);
}

function switchModeling(){
    // switch between automatic model building or manual parameter tuning
    let toggle = document.getElementById('switch');
    if (toggle.getAttribute('checked')=='true'){
        toggle.setAttribute('checked', 'false');
        params['switch']='false';
        for (id of ['ntrees', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']){
            input = document.getElementById(id);
            input.removeAttribute('disabled');
        }
    } else {
        toggle.setAttribute('checked', 'true');
        params['switch']='true';
        for (id of ['ntrees', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']){
            input = document.getElementById(id);
            input.setAttribute('disabled', null);
        }
    }

}

function updateModelBadges(){
    // update badges over model tuning controls
    input_badge_dict = {'ntrees': 'ntrees_badge', 
                        'max_depth': 'depth_badge', 
                        'min_samples_split': 'nodesize_badge', 
                        'min_samples_leaf': 'leafsize_badge', 
                        'max_features': 'maxfeats_badge'
                    }
    for (input_id in input_badge_dict){
        badge_id = input_badge_dict[input_id];
        badge = document.getElementById(badge_id);
        input_value = document.getElementById(input_id).getAttribute('value');
        badge_txt = document.createTextNode(input_value);
        badge.removeChild(badge.firstChild);
        badge.appendChild(badge_txt);
    }
}

function showSpinner(){
    let spinner = document.getElementById("spinner");
    spinner.classList.add('spinner-grow');
    let info = document.getElementById("model_info");
    while (info.firstChild) {
        info.removeChild(info.firstChild);
    }
    let loading_txt=document.createTextNode("Building model. This may take up to 30sec depending on the parameters.");
    info.appendChild(loading_txt);
}

function hideSpinner(){
    let spinner = document.getElementById("spinner");
    spinner.classList.remove('spinner-grow')
    let info = document.getElementById("model_info");
    info.removeChild(info.firstChild)
}

function showModelScores(params, train_score, test_score){
    let div = document.getElementById("model_info");
    let txt;
    // model parameters info
    if (params['switch']=='false'){
        txt = document.createTextNode("Chosen model parameters:");
    } else {
        txt = document.createTextNode("Optimal model parameters:");
    }
    div.appendChild(txt);
    div.appendChild(document.createElement("br"));
    let ul = document.createElement("ul");
    let li;
    for (key of ['ntrees', 'max_depth', 'min_samples_split',
                'min_samples_leaf', 'max_features']){
        txt = document.createTextNode(`${key}=${params[key]}`);
        li = document.createElement("li");
        li.appendChild(txt);
        ul.appendChild(li);
    }
    div.appendChild(ul);
    div.appendChild(document.createElement("br"));
    // model scores on train
    txt = document.createTextNode("Model scores on the train set:");
    div.appendChild(txt);
    div.appendChild(document.createElement("br"));
    let tbl = document.createElement("table");
    tbl.classList.add("table"); 
    tbl.classList.add("table-hover");
    let thead = document.createElement("thead");
    let tbody = document.createElement("tbody");
    let td, th, tr;
    for (entry of ["Metric", "Value"]){
        th = document.createElement("th");
        th.appendChild(document.createTextNode(entry));
        thead.appendChild(th);
    }
    tbl.appendChild(thead);
    for (key in train_score){
        tr = document.createElement("tr");
        td = document.createElement("td");
        td.appendChild( document.createTextNode(`${key}`) );
        tr.appendChild(td);
        td = document.createElement("td");
        td.appendChild( document.createTextNode(`${train_score[key].toFixed(2)}`) );
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
    tbl.appendChild(tbody);
    div.appendChild(tbl);
    div.appendChild(document.createElement("br"));
    // model scores on test
    txt = document.createTextNode("Model scores on the test set:");
    div.appendChild(txt);
    div.appendChild(document.createElement("br"));
    tbl = document.createElement("table");
    tbl.classList.add("table"); 
    tbl.classList.add("table-hover");
    thead = document.createElement("thead");
    tbody = document.createElement("tbody");
    for (entry of ["Metric", "Value"]){
        th = document.createElement("th");
        th.appendChild(document.createTextNode(entry));
        thead.appendChild(th);
    }
    tbl.appendChild(thead);
    for (key in test_score){
        tr = document.createElement("tr");
        td = document.createElement("td");
        td.appendChild( document.createTextNode(`${key}`) );
        tr.appendChild(td);
        td = document.createElement("td");
        td.appendChild( document.createTextNode(`${test_score[key].toFixed(2)}`) );
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
    tbl.appendChild(tbody);
    div.appendChild(tbl);
}

function showFigures(fig_data){
    let div = document.getElementById("data_fig");
    let tgt = fig_data.filter(data => data[2]==1);
    let non_tgt = fig_data.filter(data => data[2]==0);
    console.log(tgt);
    let trace1 = {
        x: tgt.map(item => item[0]),
        y: tgt.map(item => item[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Target',
        marker: {   size: 10,
                    opacity: .75
                }
      };
    let trace2 = {
        x: non_tgt.map(item => item[0]),
        y: non_tgt.map(item => item[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Non-target',
        marker: {   size: 10,
                    opacity: .75
                }
      };
    let traces = [trace1, trace2];
    let layout = {
        title: {
            text: 'Train set in the space of the first two PCA components'
        },
        xaxis: {
            title: { text: '1st component'}
        },
        yaxis: {
            title: { text: '2nd component'}
        }
    }
    Plotly.newPlot(div, traces, layout);
}

async function submitParams(){

    showSpinner();
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
    })
    .then(response => response.json())
    .then(data => {
        hideSpinner();
        showModelScores(data["data_params"], 
                        data["train_score"],
                        data["test_score"]);
        showFigures(data["pca_data"]);
    })
    .catch((error) => {
        console.error('Error:', error);
        hideSpinner();
        let info = document.getElementById("model_info");
        let error_txt=document.createTextNode("An error occurred during model building. Please try again.");
        info.appendChild(error_txt);
    });
    

}