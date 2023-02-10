for i in $(seq 1 1);
do
    # nvprof  --profile-from-start off --aggregate-mode off --replay-mode application --csv --log-file log_$i.csv --event-collection-mode continuous -m nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_transmit_throughput,nvlink_receive_throughput,nvlink_overhead_data_transmitted,nvlink_overhead_data_received,nvlink_total_response_data_received,nvlink_user_response_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_1.csv --event-collection-mode continuous -m nvlink_total_data_transmitted ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_2.csv --event-collection-mode continuous -m nvlink_total_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_3.csv --event-collection-mode continuous -m nvlink_transmit_throughput ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_4.csv --event-collection-mode continuous -m nvlink_receive_throughput ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_5.csv --event-collection-mode continuous -m nvlink_overhead_data_transmitted ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_6.csv --event-collection-mode continuous -m nvlink_overhead_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_7.csv --event-collection-mode continuous -m nvlink_total_response_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_8.csv --event-collection-mode continuous -m nvlink_user_response_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_9.csv --event-collection-mode continuous -m nvlink_total_write_data_transmitted ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_10.csv --event-collection-mode continuous -m nvlink_user_data_transmitted ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_11.csv --event-collection-mode continuous -m nvlink_user_data_received ./p2p
    nvprof  --profile-from-start off --aggregate-mode off --csv --log-file log_12.csv --event-collection-mode continuous -m nvlink_user_write_data_transmitted ./p2p

    
done



