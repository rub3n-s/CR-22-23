function output = GUI_trainDigits(imgPath,netPath) 

disp(['A iniciar leitura do ficheiro ', imgPath]);

% Carregar a rede previamente treinada
load(netPath, 'net');

% Obtem o input
in = binarizedImageInput(imgPath, 0);

% Simular
out = sim(net, in);

% b - index de cada array
% Cada array corresponde a um digito
[a, b] = max(out);

switch(b)
    case 1
        disp('Resultado:  0');
        output = '0';
    case 2
        disp('Resultado:  1');
        output = '1';
    case 3
        disp('Resultado:  2');
        output = '2';
    case 4
        disp('Resultado:  3');
        output = '3';
    case 5
        disp('Resultado:  4');
        output = '4';
    case 6
        disp('Resultado:  5');
        output = '5';
    case 7
        disp('Resultado:  6');
        output = '6';
    case 8
        disp('Resultado:  7');
        output = '7';
    case 9
        disp('Resultado:  8');
        output = '8';
    case 10
        disp('Resultado:  9');
        output = '9';
    otherwise
        disp('Nao foi possivel reconhecer');
        output = 'Nao foi possivel reconhecer';
end
end