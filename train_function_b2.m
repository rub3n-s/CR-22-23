function train_function_b2()
%% NOTA
%   Neste exemplo são utilizadas duas redes
%       Uma para os digitos de 0 a 9
%       Outra para os operadores
%% Definir Constantes e Variaveis
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
NUM_FILES = 50;

% Numero de pastas
NUM_DIGIT_FOLDERS = 10;
NUM_OPERATOR_FOLDERS = 4;

% Gerar uma matriz
binaryMatrixDigits = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrixDigits = [];

binaryMatrixOperators = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrixOperators = [];

%% [Digitos] 
% ======== Ler, redimensionar e preparar os targets ========
count = 1;
for i=1:NUM_DIGIT_FOLDERS
    % Definir o caminho para a pasta
    FOLDER_PATH = sprintf('NN_datasets/train/%d/',i-1);    

    % Percorrer os ficheiros (50) dentro da pasta i
    for j=1:NUM_FILES
        img = imread(strcat(FOLDER_PATH,sprintf('%d.png',j)));
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixDigits(:, count) = reshape(binarizedImg, 1, []);
        count=count+1;
    end
end

% Obter todos os vetores (cada um corresponde a uma pasta)
vec1 = repelem(1, NUM_FILES);
vec2 = repelem(2, NUM_FILES);
vec3 = repelem(3, NUM_FILES);
vec4 = repelem(4, NUM_FILES);
vec5 = repelem(5, NUM_FILES);
vec6 = repelem(6, NUM_FILES);
vec7 = repelem(7, NUM_FILES);
vec8 = repelem(8, NUM_FILES);
vec9 = repelem(9, NUM_FILES);
vec10 = repelem(10, NUM_FILES);

% Preencher a matriz a partir dos vetores
targetMatrixDigits = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, ...
                vec8, vec9, vec10];

targetDigits = onehotencode(targetMatrixDigits,1,'ClassNames',1:10);
inDigits = binaryMatrixDigits;

% ======== Treinar rede ========
% Testar com x neuronios e y camadas escondidas
netDigits = feedforwardnet([5 5]);

% ======== Configurar a Rede ========
% Função de Ativação
netDigits.layers{1}.transferFcn = 'tansig';
netDigits.layers{2}.transferFcn = 'purelin';
netDigits.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
netDigits.trainParam.epochs = 100;

% Funcao de Treino
netDigits.trainFcn = 'trainlm';

% Divisao de Treino
netDigits.divideFcn = 'dividerand';
netDigits.divideParam.trainRatio = 0.70;
netDigits.divideParam.valRatio = 0.15;
netDigits.divideParam.testRatio = 0.15;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumTrainsDigits = 0;
for k=1:10
    % Treinar 
    [netDigits,trDigits] = train(netDigits, inDigits, targetDigits);    

    % Simular
    outDigits = sim(netDigits, inDigits);
    
    r = 0;
    for i=1:size(outDigits,2)
        [a, b] = max(outDigits(:,i));
        [c, d] = max(targetDigits(:,i));
        if b == d
          r = r+1;
        end
    end
    
    accuracy = r/size(outDigits,2) * 100;
    sumTrainsDigits = sumTrainsDigits + accuracy;
    fprintf('\nPrecisao do Treino [%d] = %.2f\n', k, accuracy)

    %plotconfusion(target,out) % Matriz de confusao

    % Desempenho
    %disp(tr.performFcn);
    disp('========== Digits Performance ==========')
    disp(['Train Performance: ' num2str(trDigits.best_perf)])
    disp(['Validation Performance: ' num2str(trDigits.best_vperf)]);
    disp(['Test Performance: ' num2str(trDigits.best_tperf)]);
end

%% [Operadores] 
% ======== Ler, redimensionar e preparar os targets ========
count = 1;
for i=1:NUM_OPERATOR_FOLDERS
    % Definir o caminho para a pasta
    switch(i)
        case 1 % add
            FOLDER_PATH = 'NN_datasets/train/add/';
        case 2 % div
            FOLDER_PATH = 'NN_datasets/train/div/';
        case 3 % mul
            FOLDER_PATH = 'NN_datasets/train/mul/';
        case 4 % sub
            FOLDER_PATH = 'NN_datasets/train/sub/';
    end

    % Percorrer os 50 ficheiros dentro da pasta i
    for j=1:NUM_FILES
        img = imread(strcat(FOLDER_PATH,sprintf('%d.png',j)));
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixOperators(:, count) = reshape(binarizedImg, 1, []);
        count=count+1;
    end
end

% Obter todos os vetores (cada um corresponde a uma pasta)
vec1 = repelem(1, NUM_FILES);
vec2 = repelem(2, NUM_FILES);
vec3 = repelem(3, NUM_FILES);
vec4 = repelem(4, NUM_FILES);

% Preencher a matriz a partir dos vetores
targetMatrixOperators = [vec1, vec2, vec3, vec4];

targetOperators = onehotencode(targetMatrixOperators,1,'ClassNames',1:4);
inOperators = binaryMatrixOperators;

% ======== Treinar rede ========
% Testar com x neuronios e y camadas escondidas
netOperators = feedforwardnet([5 5]);

% ======== Configurar a Rede ========
% Função de Ativação
netOperators.layers{1}.transferFcn = 'tansig';
netOperators.layers{2}.transferFcn = 'purelin';
netOperators.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
netOperators.trainParam.epochs = 100;

% Funcao de Treino
netOperators.trainFcn = 'trainlm';

% Divisao de Treino
netOperators.divideFcn = 'dividerand';
netOperators.divideParam.trainRatio = 0.70;
netOperators.divideParam.valRatio = 0.15;
netOperators.divideParam.testRatio = 0.15;

% ======== Treinar, Simular e Apresentar Resultados ========
% Realizar 10 iteracoes de treino e calcular media
sumTrainsOperators = 0;
for k=1:10
    % Treinar 
    [netOperators,trOperators] = train(netOperators, inOperators, targetOperators);    

    % Simular
    outOperators = sim(netOperators, inOperators);
    
    r = 0;
    for i=1:size(outOperators,2)
        [a, b] = max(outOperators(:,i));
        [c, d] = max(targetOperators(:,i));
        if b == d
          r = r+1;
        end
    end
    
    accuracy = r/size(outOperators,2) * 100;
    sumTrainsOperators = sumTrainsOperators + accuracy;
    fprintf('\nPrecisao do Treino [%d] = %.2f\n', k, accuracy)

    %plotconfusion(target,out) % Matriz de confusao

    % Desempenho
    %disp(tr.performFcn);
    disp('========== Operators Performance ==========')
    disp(['Train Performance: ' num2str(trOperators.best_perf)])
    disp(['Validation Performance: ' num2str(trOperators.best_vperf)]);
    disp(['Test Performance: ' num2str(trOperators.best_tperf)]);
end

%% [Digitos] Apresentar a Media
fprintf('\n ------ [Digitos] Apos 10 iteracoes ------\n')
fprintf('Media de Precisao = %.2f\n', sumTrainsDigits/10);

%% [Operadores] Apresentar a Media
fprintf('\n ------ [Operadores] Apos 10 iteracoes ------\n')
fprintf('Media de Precisao = %.2f\n', sumTrainsOperators/10);

end