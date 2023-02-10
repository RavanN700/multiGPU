for i in $(seq 1 10);
do
    nvprof  --profile-from-start off --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_total_data_transmitted ./p2p >> output_0.csv
    nvprof  --profile-from-start off --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_total_data_received ./p2p >> output_1.csv
    nvprof  --profile-from-start off --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_transmit_throughput ./p2p >> output_2.csv
    nvprof  --profile-from-start off --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_receive_throughput ./p2p >> output_3.csv
done



