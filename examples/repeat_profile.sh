for i in $(seq 1 5);
do
    nvprof  --profile-from-start off --aggregate-mode off --replay-mode application --csv --log-file log_$i.csv --event-collection-mode continuous -m nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_transmit_throughput,nvlink_receive_throughput,nvlink_overhead_data_transmitted,nvlink_overhead_data_received,nvlink_total_response_data_received,nvlink_user_response_data_received ./p2p

done



