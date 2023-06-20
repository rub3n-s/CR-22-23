function [binaryMatrix,targetMatrix] = getBinaryMatrixTargetMatrix(DATASET_FOLDER,NUM_FOLDERS,NUM_FILES)
%% Definir Constantes
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];
count = 1;

%% Ler, redimensionar e preparar os targets
fprintf('\n===== A INICIAR LEITURA DE IMAGENS =====\nPastas acessadas:\n');
for i=1:NUM_FOLDERS
    % Apenas para debug (Mostrar o path da pasta que esta a ser acedida)
    switch(i-1)
        case 10     % add
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'add'));
        case 11     % div
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'div'));
        case 12     % mul
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'mul'));
        case 13     % sub
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,'sub'));
        otherwise   % [0,9]
            disp(fullfile('..','NN_datasets',DATASET_FOLDER,sprintf('%d',i-1)));
    end

    % Percorrer os 5 ficheiros dentro da pasta i
    for j=1:NUM_FILES
        switch(i-1)
            case 10     % add
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'add',sprintf('%d.png',j));
            case 11     % div
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'div',sprintf('%d.png',j));
            case 12     % mul
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'mul',sprintf('%d.png',j));
            case 13     % sub
                file = fullfile('..','NN_datasets',DATASET_FOLDER,'sub',sprintf('%d.png',j));
            otherwise   % [0,9]
                file = fullfile('..','NN_datasets',DATASET_FOLDER,sprintf('%d',i-1),sprintf('%d.png',j));
        end
        img = imread(file);
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrix(:, count) = reshape(binarizedImg, 1, []);
        count=count+1;
    end

    % Obter todos os vetores (cada um corresponde a uma pasta)
    % Devolve um vetor em que o primeiro parametro é um escalar (index da
    %   pasta) e o segundo parametro é o numero de elementos
    
    % Obter a matriz a partir dos vetores
    targetMatrix = [targetMatrix, repelem(i, NUM_FILES)];
end
end