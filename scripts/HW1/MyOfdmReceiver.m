
function [decoded_data] = MyOfdmReceiver(data)

    OFDM_TX;  % run transmitter code to load sts and lts and other parameters 
     
    % Rx processing params:
    rx = data;              % run OFDM tx code to get raw_rx_dec
    
    %% Packet Detection

%{
    preamble = [repmat(sts_t, 1, 30)  lts_t(33:64) lts_t lts_t];
    30 STS (sts_t) -> 2.5 LTS (lts_t) -> payload
%}

    THRESH = 0.95;
    
    % LTS:

    L = length(lts_t);
    M = zeros(1, length(rx) - L);
    for d = 1:(length(rx) - L)
        P = sum( conj(rx(d:d+L-1)) .* lts_t );  % cross corr with lts
        E1 = sum( (abs(rx(d:d+L-1))).^2 );
        E2 = sum( (abs(lts_t)).^2 );

        M(d) = abs(P)^2 / (E1*E2);      % for normalization
    end

    %[~, pktStart] = max(Mavg);
    [~, threshIdx] = find(M > 0.4, 1, 'first');  %
    pkt_start = threshIdx - (length(sts_t)*30) - (length(lts_t)/2);  % reindexing to find packet start

    fprintf("Packet start (LTS): %d\n\n", pkt_start);

    figure;
    stem(M);
    xline(threshIdx, '--r');
    ylim([0, 1.1]);
    title('Correlation using LTS');

    figure;
    stem(M);
    xline(threshIdx, '--r');
    xlim([0, 1000]); ylim([0, 1.1]);
    title('ZOOMED Correlation using LTS');

    len_lts = L;
    pkt_start_lts = pkt_start;

    % STS:
    % the idea here is to index 2 sts lengths ahead and autocorrelate
    % those windows (2 windows each with sts length = 16) with each other, indexing one by one through the rx
    % stream. in real-time scenario i get it would likely exit once a
    % threshold is reached and not index through the entire stream, but
    % this can be easily implemented
    %
    % i then normalized by using the TOTAL energy because this seemed to
    % make more sense to me, i.e., the energy in window 1 AND energy and
    % window 2. this normalization is necessary because we dont want to favor signals
    % that just so happen to have high power but little correlation

    L = length(sts_t);
    M = zeros(1, length(rx) - 2*L);
    for d = 1:(length(rx) - 2*L)
        P = sum( conj(rx(d:d+L-1)) .* rx(d+L:d+(2*L)-1) );    % correlation of adjacent STS's
        E1 = sum( (abs(rx(d:d+L-1))).^2 );                    % total energy of 1st window
        E2 = sum( (abs(rx(d+L:d+(2*L)-1))).^2 );

        M(d) = abs(P)^2 / (E1*E2);      % for normalization
    end
    
    windowSize = L * 28 + 1;

    b = (1/windowSize)*ones(1,windowSize);
    a = 1;
    Mavg = filter(b,a,M);

    %[~, pktStart] = max(Mavg);
    [~, threshIdx] = find(Mavg > THRESH, 1, 'first');
    [~, maxIdx] = max(Mavg(threshIdx:threshIdx+100));
    pkt_start = threshIdx + maxIdx - windowSize;     % reindexing to find packet start

    fprintf("Packet start (STS): %d\n\n", pkt_start);

    % plotting
    figure;
    subplot(2,1,1);
    stem(M);
    xline(pkt_start, '--r');
    ylim([0, 1.1]);
    title('Correlation using STS (not averaged)');
    subplot(2,1,2);
    stem(Mavg);
    xline(threshIdx + maxIdx - 1, '--r');
    ylim([0, 1.1]);
    title('Correlation using STS (moving average)');

    figure;
    subplot(2,1,1);
    stem(M);
    xline(pkt_start, '--r');
    xlim([0, 1000]); ylim([0, 1.1]);
    title('ZOOMED Correlation using STS (not averaged)');
    subplot(2,1,2);
    stem(Mavg);
    xline(threshIdx + maxIdx - 1, '--r');
    xlim([0, 1000]); ylim([0, 1.1]);
    title('ZOOMED Correlation using STS (moving average)');

    len_sts = L;
    pkt_start_sts = pkt_start;
    
    %pkt_start_sts = pkt_start_lts

    
    %% CFO estimation and correction

%{
    angle(rx(122))
    angle(rx(122 + 16))
    angle(rx(122 + 32))
    angle(rx(122 + 48))
%}

    % the logic is to measure the frequency difference for each index along
    % all adjacent symbols by, for each index, to take the conjugate of one
    % of the indices (in this case the earlier sts), and multiply the two
    % indices together (eg sts_1(1) * conj(sts_2(1)). the phase of this
    % would ideally equal 0, but results in (w_1 - w_2), which
    % conveys the offset. i do this for every index of two sequences, sum
    % them, take the phase, take the average, and then average all values across
    % all sts's, before taking the phase to cancel as much noise as
    % possible. i then repeat this process/fine-tune with lts.
    %
    % *the sum of the offset/drift/cfo phase cannot exceed pi to be
    % deterministic (similar to aliasing), so hopefully cfo would have
    % already been corrected enough from the previous sts correction

    % STS CFO:
    avg_sts_offset = zeros(1, 29);

    for sts_num = 1:29  % looks one sts seq ahead, so 30-1

        sts_start = pkt_start_sts + (len_sts*(sts_num-1));
        sts_end = sts_start + len_sts - 1;
        
        next_sts_start = sts_end + 1;
        next_sts_end = next_sts_start + len_sts - 1;

        avg_sts_offset(sts_num) = sum(conj(rx(sts_start:sts_end)) .* rx(next_sts_start:next_sts_end));
    end
    
    sts_cfo = angle(sum(avg_sts_offset)) / 16;  % calculate average cfo per sample

    % apply sts (coarse) cfo
    rx_sts_cfo = rx .* exp(-1j * sts_cfo * (0:length(rx)-1));

    % LTS CFO (follows exact same logic as above)
    lts_start = pkt_start_sts + (30*len_sts + 32);  % accounting for half lts before
    lts_end = lts_start + len_lts - 1;
    
    next_lts_start = lts_end + 1;
    next_lts_end = next_lts_start + len_lts - 1;

    avg_lts_offset = sum(conj(rx_sts_cfo(lts_start:lts_end)) .* rx_sts_cfo(next_lts_start:next_lts_end));
    %avg_lts_offset = sum(conj(rx(lts_start:lts_end)) .* rx(next_lts_start:next_lts_end));
    %avg_lts_offset = sum(conj(rx(lts_start:lts_end)) .* rx(lts_start:lts_end));

    lts_cfo = angle(sum(avg_lts_offset)) / 64;

    % apply lts (fine) cfo
    rx_cfo = rx_sts_cfo .* exp(-1j * lts_cfo * (0:length(rx)-1));
    %rx_cfo = rx .* exp(-1j * lts_cfo * (0:length(rx)-1));

    %% Channel estimation and correction
    % Use the two copies of LTS and find channel estimate (Reference: Thesis)
    % Convert channel estimate to matrix form and equalize the above matrix
    % Output : Symbol equalized matrix in frequency domain of same size

    % ESTIMATION:
    % cross correlation would result in (h[n]*x[n])*x[-n] = h[n]*autocorr[n]
    % (where * is convolution), which does not isolate the channel UNLESS the autocorr
    % was an impulse. if you wanted to stay in the time domain you would
    % have to instead do conj(x[n])(h[n]*x[n]) / |x[n]|^2, which cancels
    % the phase in the numerator before normalizing
    % 
    % you could also realize that, since the channel
    % acts as a convolution distortion to the signal, it is equivalent to first convert to
    % the frequency domain and divide by the known preamble, i.e.,
    % X[k]H[k] = Y[k], so Y[k]/X[k] = H[k], where X[k] is the preamble. so thats what i did
    %
    % also the first 32 bits of the ltf acts as a CP and the channel
    % estimate has to equal the length of the ofdm symbol payload for
    % equalization to mathematically work here. so we discard the first 32
    % bits and do this to 2 (both) lts symbols and average them after,
    % which also makes it more robust to noise since its independent and 
    % random (i.e., incoherent)

    % EQUALIZATION:
    % for each ofdm symbol you would then remove the 16 bit cp so youre
    % left with a 64 bit payload, and after a 64-point fft (which is big
    % enough since payload consists of some null subcarrriers so
    % 64 > important subarriers) you can divide by H[k]
    

    % CHANNEL ESTIMATION:
    lts1_t = rx_cfo(lts_start : lts_end);
    lts2_t = rx_cfo(next_lts_start : next_lts_end);

    %lts_avg_f = (fft(lts1_t) + fft(lts1_t)) / 2;
    lts_avg_f = (fft(lts1_t) + fft(lts2_t)) / 2;  % averages two freq domain lts seq

    H = lts_avg_f ./ lts_f;  % estimates channel (Y[k]/X[k])

    
    %% Payload Processing (CP Removal, FFT, etc.)

    % refer to above for equalization notes

%{
    SC_IND_DATA = [2:7 9:21 23:27 39:43 45:57 59:64];
%}
    
    data_start = next_lts_end + 1;

    payload_matrix = zeros(N_SC, N_OFDM_SYMS);  % (64 subcarriers x 500 symbols)
    
    for i = 1:N_OFDM_SYMS
        ofdm_sym_idx = data_start + (i-1)*(N_SC + CP_LEN);  % N_SC + CP_LEN = 80

        payload_t = rx_cfo(ofdm_sym_idx + CP_LEN : ofdm_sym_idx + CP_LEN + N_SC - 1);
        payload_f = fft(payload_t);

        payload_equalized = payload_f ./ H;
        %payload_equalized = payload_f .* conj(H);

        payload_matrix(:, i) = payload_equalized.';  % transform row to col vector and store in matrix for later
    end

    %rx_syms = payload_matrix(SC_IND_DATA, :);
    %rx_syms = rx_syms(:).';

    %% SFO estimation and correction using pilots
    % SFO manifests as a frequency-dependent phase whose slope increases
    % over time as the Tx and Rx sample streams drift apart from one
    % another. To correct for this effect, we calculate this phase slope at
    % each OFDM symbol using the pilot tones and use this slope to
    % interpolate a phase correction for each data-bearing subcarrier.
    % Output: Symbol equalized matrix with pilot phase correction applies
    
    %% Phase Error Correction using pilots
    % Extract the pilots and calculate per-symbol phase error
    % Output: Symbol equalized matrix with pilot phase correction applied
    % Remove pilots and flatten the matrix to a vector rx_syms


    % these both are related to each other can be fixed simultaneously:
    %
    % we predict X[k] as Y[k]H*[k] already, meaning our received pilot
    % symbol can be represented as P_rec = Y[k]H*[k].
    % similar logic can be applied as CFO: we first do (P_rec)(P_orig)*,
    % and find the phase (w_rec - w_orig)(n).
    % now since we have multiple pilots, we essentially have points on
    % a line in the form (n, phase). if we then draw a line of best fit,
    % it will be in the form: phase = Mn + C.
    %
    % SFO estimation is M and the phase error is C.
    %
    % this line would be found using least squares but matlab has a function called polyfit for this
    % we finally apply the correction via (Y)(exp(-j(Mn+C))

%{
    SC_IND_PILOTS = [8 22 44 58];

    % Define the pilot tone values as BPSK symbols
    pilots = [1 1 -1 1].';
    
    % Repeat the pilots across all OFDM symbols
    pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);
    
    *802.11 defines pilots as indices -21 -7 7 21

    "p = polyfit(x,y,n) returns the coefficients for a polynomial p(x) of 
    degree n that is a best fit (in a least-squares sense) for the data in 
    y. The coefficients in p are in descending powers, and the length of p 
    is n+1"
%}
    % Define the pilot tone values as BPSK symbols
    pilots = [1 1 -1 1].';
    
    % Repeat the pilots across all OFDM symbols
    pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);

    k_vals = [-(N_SC/2) : N_SC/2 - 1].';    % col vector from -32 to 31
    k_shifted = ifftshift(k_vals);          % have to turn [-32 ... 31] to [0 ... 31 -32 ... -1] to fit fft format
    
    k_pilots = k_shifted(SC_IND_PILOTS);
    %k_pilots = k_vals(SC_IND_PILOTS)

    for i = 1:N_OFDM_SYMS
        pilots_val = payload_matrix(SC_IND_PILOTS, i);  % row, col
        pilots_phase = angle(pilots_val .* conj(pilots_mat(:, i)));
        
        [k_vals_ordered, idx] = sort(k_pilots);
        pilots_phase_ordered = pilots_phase(idx);  % extract phases from pilot indices

        coeffs = polyfit(k_vals_ordered, pilots_phase_ordered, 1);  % returns slope val then const val

        slope = coeffs(1);
        const = coeffs(2);

        payload_matrix(:, i) = payload_matrix(:, i) .* exp(-1j * (slope * k_shifted + const));
    end

    rx_syms = payload_matrix(SC_IND_DATA, :);
    rx_syms = rx_syms(:).';
   
    
    %% Demodulation
    
    figure;
    scatter(real(rx_syms), imag(rx_syms),'filled');
    title('Signal Space of RECEIVED bits');
    xlabel('I'); ylabel('Q');
    
    % FEC decoder
    Demap_out = demapper(rx_syms,MOD_ORDER,1);
    
    % viterbi decoder
    decoded_data = vitdec(Demap_out,trel,7,'trunc','hard');
    
    % decoded_data is the final output corresponding to tx_data, which can be used
    % to calculate BER

end
