clc;
clear all;
close all;

T=2;
S=5;
P=4;
N_hat=P*S;

K_hat=4;
M_hat=4;


vt=K_hat/S;
vf=M_hat/P;

K=floor(N_hat/M_hat);
M=floor(N_hat/K_hat);
N=K*M;

vf_real=S/K;
vt_real=P/M;


o=ones(1, P);
z=zeros(1, P*S-P);
Gf=[o z];
Gf=circshift(Gf, [0, -floor(P/2)]);
g=ifft(Gf);


g = sqrt(vt_real*vf_real)*g/ sqrt(sum(abs(g).^2));

E=vf_real*vt_real;

for m = 1:M
    
    p = mod((m-1)*(vt*S),N_hat);
    for k = 1:K
        
        A(:,(m-1)*K+(k-1)+1) = circshift(g,[0 p]).*exp((0:N_hat-1)*1j*2*pi*(k-1)*vf/S);
        
    end
end

% Preparação dataset 
num_samples_per_snr = 50000;
SNR_range = 0:1:10;
num_snr = length(SNR_range);
total_samples = num_samples_per_snr * num_snr;

% Pré-alocação para performance
r_data = zeros(total_samples, N);
s_data = zeros(total_samples, N);
snr_labels = zeros(total_samples, 1);

idx = 1;

%h1=[(exp(0)) (exp(-3)) (exp(-6)) (exp(-9))];
h1 = [1 0.2 0.1 0.04];
h1=[h1 zeros(1,N_hat-length(h1))];
H1=zeros(N_hat,N_hat);
for i_CH1=1:(N_hat)
    H1(:,i_CH1)= circshift(h1,[0 (i_CH1-1)]);
end

for snr_idx = 1:num_snr
    SNR_dB = SNR_range(snr_idx);
    SNR_L = 10^(SNR_dB/10);
    sigma = sqrt(E/(2*SNR_L));
    
    for sample = 1:num_samples_per_snr
        d = randi([0 1], N, 1);
        s = qammod(d, 2);
        
        x = A * s;
        w = sigma*(randn(size(x)) + 1j*randn(size(x)));
        
        y = H1*x + w;

        yeq=ifft(fft(y)./(fft(H1(:,1))));
        
        r = A' * yeq;
        
        r_data_real(idx, :) = real(r).';
        r_data_imag(idx, :) = imag(r).';
        s_data(idx, :) = s.';
        snr_labels(idx) = SNR_dB;
        
        idx = idx + 1;
    end
end

% Criar tabela e salvar
r_real_names = arrayfun(@(x) sprintf('r_re_%d', x), 1:N, 'UniformOutput', false);
r_imag_names = arrayfun(@(x) sprintf('r_im_%d', x), 1:N, 'UniformOutput', false);
s_names = arrayfun(@(x) sprintf('s_%d', x), 1:N, 'UniformOutput', false);

data_table = array2table([snr_labels, r_data_real, r_data_imag, s_data]);
data_table.Properties.VariableNames = ['SNR_dB', r_real_names, r_imag_names, s_names];

writetable(data_table, ['dataset_FTNGFDM_SNR_TIFS_50000_Treino.csv']);

disp('Dataset corrigido e salvo com sucesso em dataset_FTNGFDM_SNR_TIFS_50000_Teste.csv');
