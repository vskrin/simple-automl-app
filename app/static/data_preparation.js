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

async function submitParams(){

    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
    })
    .then(response => response.json())
    .then(data => {
        model = JSON.stringify(data);
        console.log('model: ', model);
    })
    .catch((error) => {
        console.error('Error:', error);
    });

}