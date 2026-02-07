%% Determine MIMO Configuration
% Check which variables exist to determine the configuration
if exist('rx_vec_dec_1A', 'var')
    % Part 2: 1x4 MIMO (1 Tx, 4 Rx)
    NUM_RX = 4;
    NUM_TX = 1;
    raw_rx_dec_A = rx_vec_dec_1A;
    raw_rx_dec_B = rx_vec_dec_1B;
    raw_rx_dec_C = rx_vec_dec_1C;
    raw_rx_dec_D = rx_vec_dec_1D;
    fprintf('Configuration: 1x4 MIMO (1 Tx, 4 Rx)\n');
elseif exist('raw_rx_dec_2A', 'var')
    % Part 3: 2x2 MIMO, 1 spatial stream
    NUM_RX = 2;
    NUM_TX = 1;
    raw_rx_dec_A = raw_rx_dec_2A;
    raw_rx_dec_B = raw_rx_dec_2B;
    fprintf('Configuration: 2x2 MIMO, 1 spatial stream\n');
elseif exist('raw_rx_dec_3A', 'var')
    % Part 4: 2x2 MIMO, 2 spatial streams
    NUM_RX = 2;
    NUM_TX = 2;
    raw_rx_dec_A = raw_rx_dec_3A;
    raw_rx_dec_B = raw_rx_dec_3B;
    fprintf('Configuration: 2x2 MIMO, 2 spatial streams\n');
else
    error('No valid received signal variables found');
end

trel = poly2trellis(7, [171 133]);

%% Correlate for LTS
LTS_CORR_THRESH=.4;
DO_APPLY_CFO_CORRECTION=1;
DO_APPLY_SFO_CORRECTION=1;
DO_APPLY_PHASE_ERR_CORRECTION=1;

% For simplicity, we'll only use RFA for LTS correlation and peak
% discovery. A straightforward addition would be to repeat this process for
% RFB and combine the results for detection diversity.

% Complex cross correlation of Rx waveform with time-domain LTS
lts_corr = abs(conv(conj(fliplr(lts_t)), sign(raw_rx_dec_A)));

% Skip early and late samples - avoids occasional false positives from pre-AGC samples
lts_corr = lts_corr(32:end-32);

% Find all correlation peaks
lts_peaks = find(lts_corr > LTS_CORR_THRESH*max(lts_corr));

% Select best candidate correlation peak as LTS-payload boundary
% In this MIMO example, we actually have 3 LTS symbols sent in a row.
% The first two are sent by RFA on the TX node and the last one was sent
% by RFB. We will actually look for the separation between the first and the
% last for synchronizing our starting index.

[LTS1, LTS2] = meshgrid(lts_peaks,lts_peaks);
[lts_last_peak_index,y] = find(LTS2-LTS1 == length(lts_t));

% Stop if no valid correlation peak was found
if(isempty(lts_last_peak_index))
    fprintf('No LTS Correlation Peaks Found!\n');
    return;
end

% Set the sample indices of the payload symbols and preamble
% The "+32" here corresponds to the 32-sample cyclic prefix on the preamble LTS
% The "+192" corresponds to the length of the extra training symbols for MIMO channel estimation
mimo_training_ind = lts_peaks(max(lts_last_peak_index)) + 32;
payload_ind = mimo_training_ind + 192;

% Subtract of 2 full LTS sequences and one cyclic prefixes
% The "-160" corresponds to the length of the preamble LTS (2.5 copies of 64-sample LTS)
lts_ind = mimo_training_ind-160;

if(DO_APPLY_CFO_CORRECTION)
    %Extract LTS (not yet CFO corrected)
    rx_lts = raw_rx_dec_A(lts_ind : lts_ind+159); %Extract the first two LTS for CFO
    rx_lts1 = rx_lts(-64 + [97:160]);
    rx_lts2 = rx_lts( [97:160]);

    %Calculate coarse CFO est
    rx_cfo_est_lts = mean(unwrap(angle(rx_lts2 .* conj(rx_lts1))));
    rx_cfo_est_lts = rx_cfo_est_lts/(2*pi*64);
else
    rx_cfo_est_lts = 0;
end

% Apply CFO correction to raw Rx waveforms
rx_cfo_corr_t = exp(-1i*2*pi*rx_cfo_est_lts*[0:length(raw_rx_dec_A)-1]);
rx_dec_cfo_corr_A = raw_rx_dec_A .* rx_cfo_corr_t;
rx_dec_cfo_corr_B = raw_rx_dec_B .* rx_cfo_corr_t;
if NUM_RX == 4
    rx_cfo_corr_t_C = exp(-1i*2*pi*rx_cfo_est_lts*[0:length(raw_rx_dec_C)-1]);
    rx_cfo_corr_t_D = exp(-1i*2*pi*rx_cfo_est_lts*[0:length(raw_rx_dec_D)-1]);
    rx_dec_cfo_corr_C = raw_rx_dec_C .* rx_cfo_corr_t_C;
    rx_dec_cfo_corr_D = raw_rx_dec_D .* rx_cfo_corr_t_D;
end


% MIMO Channel Estimatation
lts_ind_TXA_start = mimo_training_ind + 32 ;
lts_ind_TXA_end = lts_ind_TXA_start + 64 - 1;

lts_ind_TXB_start = mimo_training_ind + 32 + 64 + 32 ;
lts_ind_TXB_end = lts_ind_TXB_start + 64 - 1;

% Extract training sequences for each Rx antenna
rx_lts_AA = rx_dec_cfo_corr_A( lts_ind_TXA_start:lts_ind_TXA_end );
rx_lts_BA = rx_dec_cfo_corr_A( lts_ind_TXB_start:lts_ind_TXB_end );

rx_lts_AB = rx_dec_cfo_corr_B( lts_ind_TXA_start:lts_ind_TXA_end );
rx_lts_BB = rx_dec_cfo_corr_B( lts_ind_TXB_start:lts_ind_TXB_end );

if NUM_RX == 4
    rx_lts_AC = rx_dec_cfo_corr_C( lts_ind_TXA_start:lts_ind_TXA_end );
    rx_lts_BC = rx_dec_cfo_corr_C( lts_ind_TXB_start:lts_ind_TXB_end );
    
    rx_lts_AD = rx_dec_cfo_corr_D( lts_ind_TXA_start:lts_ind_TXA_end );
    rx_lts_BD = rx_dec_cfo_corr_D( lts_ind_TXB_start:lts_ind_TXB_end );
end

% Convert to frequency domain
rx_lts_AA_f = fft(rx_lts_AA);
rx_lts_BA_f = fft(rx_lts_BA);

rx_lts_AB_f = fft(rx_lts_AB);
rx_lts_BB_f = fft(rx_lts_BB);

if NUM_RX == 4
    rx_lts_AC_f = fft(rx_lts_AC);
    rx_lts_BC_f = fft(rx_lts_BC);
    
    rx_lts_AD_f = fft(rx_lts_AD);
    rx_lts_BD_f = fft(rx_lts_BD);
end

%% Perform Channel estimation 

rx_H_est_AA = rx_lts_AA_f ./ lts_f;
rx_H_est_BA = rx_lts_BA_f ./ lts_f;
rx_H_est_AB = rx_lts_AB_f ./ lts_f;
rx_H_est_BB = rx_lts_BB_f ./ lts_f;

if NUM_RX == 4
    rx_H_est_AC = rx_lts_AC_f ./ lts_f;
    rx_H_est_BC = rx_lts_BC_f ./ lts_f;
    
    rx_H_est_AD = rx_lts_AD_f ./ lts_f;
    rx_H_est_BD = rx_lts_BD_f ./ lts_f;
end

%

%% Rx payload processing, Perform combining for 1X4 and 2X2 separately  

% Calculate how many complete OFDM symbols we can extract from available samples
available_payload_samples = length(rx_dec_cfo_corr_A) - payload_ind + 1;
N_OFDM_SYMS_RX = floor(available_payload_samples / (N_SC+CP_LEN));

if N_OFDM_SYMS_RX < N_OFDM_SYMS
    fprintf('Note: Extracting %d OFDM symbols (expected %d)\n', N_OFDM_SYMS_RX, N_OFDM_SYMS);
end

% Use the minimum of transmitted and extractable symbols
N_OFDM_SYMS_ACTUAL = min(N_OFDM_SYMS, N_OFDM_SYMS_RX);
payload_length_needed = N_OFDM_SYMS_ACTUAL*(N_SC+CP_LEN);
payload_end_ind = payload_ind + payload_length_needed - 1;

% Recreate pilots matrix with actual number of symbols
pilots_mat_A = repmat(pilots_A, 1, N_OFDM_SYMS_ACTUAL);

% Extract the payload samples (integral number of OFDM symbols following preamble)
payload_samples_A = rx_dec_cfo_corr_A(payload_ind : payload_end_ind);
payload_samples_B = rx_dec_cfo_corr_B(payload_ind : payload_end_ind);
payload_mat_A = reshape(payload_samples_A, (N_SC+CP_LEN), N_OFDM_SYMS_ACTUAL);
payload_mat_B = reshape(payload_samples_B, (N_SC+CP_LEN), N_OFDM_SYMS_ACTUAL);

if NUM_RX == 4
    payload_samples_C = rx_dec_cfo_corr_C(payload_ind : payload_end_ind);
    payload_samples_D = rx_dec_cfo_corr_D(payload_ind : payload_end_ind);
    payload_mat_C = reshape(payload_samples_C, (N_SC+CP_LEN), N_OFDM_SYMS_ACTUAL);
    payload_mat_D = reshape(payload_samples_D, (N_SC+CP_LEN), N_OFDM_SYMS_ACTUAL);
end


% Remove the cyclic prefix
payload_mat_noCP_A = payload_mat_A(CP_LEN+[1:N_SC], :);
payload_mat_noCP_B = payload_mat_B(CP_LEN+[1:N_SC], :);

if NUM_RX == 4
    payload_mat_noCP_C = payload_mat_C(CP_LEN+[1:N_SC], :);
    payload_mat_noCP_D = payload_mat_D(CP_LEN+[1:N_SC], :);
end

% Take the FFT
syms_f_mat_A = fft(payload_mat_noCP_A, N_SC, 1);
syms_f_mat_B = fft(payload_mat_noCP_B, N_SC, 1);

if NUM_RX == 4
    syms_f_mat_C = fft(payload_mat_noCP_C, N_SC, 1);
    syms_f_mat_D = fft(payload_mat_noCP_D, N_SC, 1);
end


%*This is optional -- SFO correction*

% Equalize pilots
% Because we only used Tx RFA to send pilots, we can do SISO equalization
% here. This is zero-forcing (just divide by chan estimates)
syms_eq_mat_pilots = syms_f_mat_A ./ repmat(rx_H_est_AA.', 1, N_OFDM_SYMS_ACTUAL);

if DO_APPLY_SFO_CORRECTION
    % SFO manifests as a frequency-dependent phase whose slope increases
    % over time as the Tx and Rx sample streams drift apart from one
    % another. To correct for this effect, we calculate this phase slope at
    % each OFDM symbol using the pilot tones and use this slope to
    % interpolate a phase correction for each data-bearing subcarrier.

	% Extract the pilot tones and "equalize" them by their nominal Tx values
    
    pilots_f_mat_sfo = syms_eq_mat_pilots(SC_IND_PILOTS, :);
    pilot_phases_sfo = angle(pilots_f_mat_sfo .* conj(pilots_mat_A));
    
    k_vals = [-(N_SC/2) : N_SC/2 - 1].';
    k_shifted = ifftshift(k_vals);
    k_pilots = k_shifted(SC_IND_PILOTS);
    
    pilot_phase_sfo_corr = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
    for i = 1:N_OFDM_SYMS_ACTUAL
        [k_vals_ordered, idx] = sort(k_pilots);
        pilots_phase_ordered = pilot_phases_sfo(idx, i);
        coeffs = polyfit(double(k_vals_ordered), double(pilots_phase_ordered), 1);
        pilot_phase_sfo_corr(:, i) = coeffs(1) * k_shifted;
    end
    
    syms_f_mat_A = syms_f_mat_A .* exp(-1j * pilot_phase_sfo_corr);
    syms_f_mat_B = syms_f_mat_B .* exp(-1j * pilot_phase_sfo_corr);
    if NUM_RX == 4
        syms_f_mat_C = syms_f_mat_C .* exp(-1j * pilot_phase_sfo_corr);
        syms_f_mat_D = syms_f_mat_D .* exp(-1j * pilot_phase_sfo_corr);
    end
    syms_eq_mat_pilots = syms_f_mat_A ./ repmat(rx_H_est_AA.', 1, N_OFDM_SYMS_ACTUAL);

	% Calculate the phases of every Rx pilot tone
 

	% Calculate the SFO correction phases for each OFDM symbol

    % Apply the pilot phase correction per symbol

else
	% Define an empty SFO correction matrix (used by plotting code below)
    pilot_phase_sfo_corr = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
end

%*This is optional* 
% Extract the pilots and calculate per-symbol phase error
if DO_APPLY_PHASE_ERR_CORRECTION
    pilots_f_mat = syms_eq_mat_pilots(SC_IND_PILOTS, :);
    pilot_phase_err = angle(mean(pilots_f_mat.*pilots_mat_A));
else
	% Define an empty phase correction vector (used by plotting code below)
    pilot_phase_err = zeros(1, N_OFDM_SYMS_ACTUAL);
end
pilot_phase_corr = repmat(exp(-1i*pilot_phase_err), N_SC, 1);

% Apply pilot phase correction to all received streams
syms_f_mat_pc_A = syms_f_mat_A .* pilot_phase_corr;
syms_f_mat_pc_B = syms_f_mat_B .* pilot_phase_corr;
if NUM_RX == 4
    syms_f_mat_pc_C = syms_f_mat_C .* pilot_phase_corr;
    syms_f_mat_pc_D = syms_f_mat_D .* pilot_phase_corr;
end

%% Perform Zero-Forcing equalization for different MIMO configurations

if NUM_RX == 4 && NUM_TX == 1
    % 1x4 MIMO: Maximum Ratio Combining (special case of Zero Forcing for 1 Tx)
    % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
    % For 1 Tx: x_hat = (H^H * y) / (H^H * H)
    
    syms_eq_mat_A = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
    
    for k = 1:N_SC
        % Channel vector for subcarrier k (4x1 vector)
        H_k = [rx_H_est_AA(k); rx_H_est_AB(k); rx_H_est_AC(k); rx_H_est_AD(k)];
        
        % Received vector for subcarrier k (4xN_OFDM_SYMS_ACTUAL matrix)
        y_k = [syms_f_mat_pc_A(k, :); 
               syms_f_mat_pc_B(k, :); 
               syms_f_mat_pc_C(k, :); 
               syms_f_mat_pc_D(k, :)];
        
        % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
        x_hat_k = (H_k' * H_k) \ (H_k' * y_k);
        syms_eq_mat_A(k, :) = x_hat_k;
    end
    
    payload_syms_mat_A = syms_eq_mat_A(SC_IND_DATA, :);
    
elseif NUM_RX == 2 && NUM_TX == 1
    % 2x2 MIMO, 1 spatial stream
    % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
    
    syms_eq_mat_A = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
    
    for k = 1:N_SC
        % Channel vector for subcarrier k (2x1 vector)
        H_k = [rx_H_est_AA(k); rx_H_est_AB(k)];
        
        % Received vector for subcarrier k (2xN_OFDM_SYMS_ACTUAL matrix)
        y_k = [syms_f_mat_pc_A(k, :); syms_f_mat_pc_B(k, :)];
        
        % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
        x_hat_k = (H_k' * H_k) \ (H_k' * y_k);
        syms_eq_mat_A(k, :) = x_hat_k;
    end
    
    payload_syms_mat_A = syms_eq_mat_A(SC_IND_DATA, :);
    
elseif NUM_RX == 2 && NUM_TX == 2
    % 2x2 MIMO, 2 spatial streams
    % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
    
    syms_eq_mat_A = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
    syms_eq_mat_B = zeros(N_SC, N_OFDM_SYMS_ACTUAL);
    
    for k = 1:N_SC
        % Channel matrix for subcarrier k (2x2 matrix)
        H_k = [rx_H_est_AA(k), rx_H_est_BA(k); ...
               rx_H_est_AB(k), rx_H_est_BB(k)];
        
        % Received vector for subcarrier k (2xN_OFDM_SYMS_ACTUAL matrix)
        y_k = [syms_f_mat_pc_A(k, :); syms_f_mat_pc_B(k, :)];
        
        % Zero Forcing: x_hat = (H^H * H)^(-1) * H^H * y
        x_hat_k = (H_k' * H_k) \ (H_k' * y_k);
        syms_eq_mat_A(k, :) = x_hat_k(1, :);
        syms_eq_mat_B(k, :) = x_hat_k(2, :);
    end
    
    payload_syms_mat_A = syms_eq_mat_A(SC_IND_DATA, :);
    payload_syms_mat_B = syms_eq_mat_B(SC_IND_DATA, :);
end

%% perform demodulate or demapping post combined symbols 
rx_syms_case_1 = reshape(payload_syms_mat_A, 1, []);

if NUM_TX == 2
    rx_syms_case_2 = reshape(payload_syms_mat_B, 1, []);
end



% plot the demodulated output rx_syms_case_1 and rx_syms_case_2
figure(4);
scatter(real(rx_syms_case_1), imag(rx_syms_case_1),'filled');
title('Signal Space of received bits - Stream 1');
xlabel('I'); ylabel('Q');
grid on;

if NUM_TX == 2
    figure(5);
    scatter(real(rx_syms_case_2), imag(rx_syms_case_2),'filled');
    title('Signal Space of received bits - Stream 2');
    xlabel('I'); ylabel('Q');
    grid on;
end


% FEC decoder for the rx_syms_case_1 and rx_syms_case_2

Demap_out_case_1 = demapper(rx_syms_case_1,MOD_ORDER,1);

% viterbi decoder
rx_data_final_1= vitdec(Demap_out_case_1,trel,7,'trunc','hard');

fprintf('TX data length: %d bits\n', length(tx_data_a));
fprintf('RX data length: %d bits\n', length(rx_data_final_1));

if NUM_TX == 2
    Demap_out_case_2 = demapper(rx_syms_case_2,MOD_ORDER,1);
    rx_data_final_2 = vitdec(Demap_out_case_2,trel,7,'trunc','hard');
end

% rx_data is the final output corresponding to tx_data, which can be used
% to calculate BER

% Since we may have extracted fewer symbols than transmitted, 
% compare only the overlapping bits
min_len_1 = min(length(tx_data_a), length(rx_data_final_1));
[number_1,ber_1] = biterr(tx_data_a(1:min_len_1), rx_data_final_1(1:min_len_1));
fprintf('BER for Stream 1: %f (comparing %d bits)\n', ber_1, min_len_1);

if NUM_TX == 2
    min_len_2 = min(length(tx_data_b), length(rx_data_final_2));
    [number_2,ber_2] = biterr(tx_data_b(1:min_len_2), rx_data_final_2(1:min_len_2));
    fprintf('BER for Stream 2: %f (comparing %d bits)\n', ber_2, min_len_2);
end