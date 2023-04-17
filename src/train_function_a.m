function train_function_a()
%% Definir Constantes
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
NUM_FILES = 5;

% Numero de pastas
NUM_FOLDERS = 14;

% Gerar uma matriz
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);
targetMatrix = [];
count = 1;

%% Ler, redimensionar e preparar os targets
for i=1:NUM_FOLDERS
    % Definir o caminho para a pasta
    switch(i)
        case {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            FOLDER_PATH = sprintf('../NN_datasets/start/%d/',i);
        case 10 % add
            FOLDER_PATH = '../NN_datasets/start/add/';
        case 11 % div
            FOLDER_PATH = '../NN_datasets/start/div/';
        case 12 % mul
            FOLDER_PATH = '../NN_datasets/start/mul/';
        case 13 % sub
            FOLDER_PATH = '../NN_datasets/start/sub/';
    end

    % Percorrer os 5 ficheiros dentro da pasta i
    for j=1:NUM_FILES
        img = imread(strcat(FOLDER_PATH,sprintf('%d.png',j)));
        img = im2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrix(:, count) = reshape(binarizedImg, 1, []);
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
vec11 = repelem(11, NUM_FILES);
vec12 = repelem(12, NUM_FILES);
vec13 = repelem(13, NUM_FILES);
vec14 = repelem(14, NUM_FILES);

% Obter a matriz a partir dos vetores
targetMatrix = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, ...
                vec8, vec9, vec10, vec11, vec12, vec13, vec14];

target = onehotencode(targetMatrix,1,'ClassNames',1:14);
in = binaryMatrix;

%% Treinar rede
% Testar com 10 camadas escondidas
net = feedforwardnet(10);

%% Realizar 10 iteracoes de treino e calcular media
somaTreinos = 0;

for k=1:10
    %% Treinar, Simular e Apresentar Resultados
    % Treinar 
    [net,tr] = train(net, in, target);    

    % Simular
    out = sim(net, in);
    
    r = 0;
    for i=1:size(out,2)
        [a b] = max(out(:,i));
        [c d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    accuracy = r/size(out,2) * 100;
    somaTreinos = somaTreinos + accuracy;
    fprintf('\nPrecisao do Treino [%d] = %.2f\n', k, accuracy)

    %plotconfusion(numTarget,out) % Matriz de confusao

    %% Desempenho
    %disp(tr.performFcn);
    disp(['Train Performance: ' num2str(tr.best_perf)])
    disp(['Validation Performance: ' num2str(tr.best_vperf)]);
    disp(['Test Performance: ' num2str(tr.best_tperf)]);
end

%% Apresentar a Media
fprintf('\n------ Apos 10 iteracoes ------\n')
fprintf('Media de Precisao = %.2f\n',somaTreinos/10);

end