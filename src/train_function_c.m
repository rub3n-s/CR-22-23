function train_function_c()
%% Definir Constantes
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
NUM_FILES = 3;

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
            FOLDER_PATH = sprintf('../NN_datasets/custom/%d/',i);
        case 10 % add
            FOLDER_PATH = '../NN_datasets/custom/add/';
        case 11 % div
            FOLDER_PATH = '../NN_datasets/custom/div/';
        case 12 % mul
            FOLDER_PATH = '../NN_datasets/custom/mul/';
        case 13 % sub
            FOLDER_PATH = '../NN_datasets/custom/sub/';
    end

    % Percorrer os 3 ficheiros dentro da pasta i
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

%% Configurar a Rede
% Função de Ativação
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
%net.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
net.trainParam.epochs = 100;

% Funcao de Treino
net.trainFcn = 'trainlm';

% Divisao de Treino
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

%% Realizar 10 iteracoes de treino e calcular media
sumGlobal = 0;
sumTest = 0;

netGlobal = 0;
netTest = 0;

for k=1:10
    %% Treinar, Simular e Apresentar Resultados
    % Treinar 
    [net,tr] = train(net, in, target);    

    % Simular
    out = sim(net, in);
    
    r = 0;
    for i=1:size(out,2)
        [a, b] = max(out(:,i));
        [c, d] = max(target(:,i));
        if b == d
          r = r+1;
        end
    end
    
    globalAccuracy = r/size(out,2)*100;
    sumGlobal= sumGlobal + globalAccuracy;
    fprintf('\nPrecisao Global = %.2f\n', k, globalAccuracy)

    % SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
    TInput = in(:, tr.testInd);
    TTargets = target(:, tr.testInd);
    
    out = sim(net, TInput);
    
    % erro = perform(net, out,TTargets);
    % fprintf('Erro na classificação do conjunto de teste %f\n', erro)
    
    %Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste
    r=0;
    for i=1:size(tr.testInd,2)        % Para cada classificacao  
      [a b] = max(out(:,i));          % b guarda a linha onde encontrou valor mais alto da saida obtida
      [c d] = max(TTargets(:,i));     % d guarda a linha onde encontrou valor mais alto da saida desejada
      if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
          r = r+1;
      end
    end

    testAccuracy = r/size(tr.testInd,2)*100;
    sumTest= sumTest + testAccuracy;
    fprintf('Precisao Teste = %.2f\n', k, testAccuracy);    

    %% Guardar Valores da Net
    % Guardar a primeira iteracao ou a net que tiver melhores valores de
    % precisao
    if (k == 1 || globalAccuracy > netGlobal && testAccuracy > netTest)
        netGlobal = globalAccuracy;
        netTest = testAccuracy;
        netAux = net; 
    end 

    %plotconfusion(target,out) % Matriz de confusao
end

%% Guardar a rede
str = strcat("..\\networks\\", 'net_c'); % substituir 'net' por uma variavel generica
save(str, 'netAux');

%% Apresentar a Media
fprintf('\nApos 10 Iterações:\n')
fprintf('\tMedia Precisao Total = %.2f\n', sumGlobal/10);
fprintf('\tMedia Precisao Teste = %.2f\n', sumTest/10);

end