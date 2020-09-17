from metrics import rouge_metric

file_reference = 'teste.txt'
reference = 'meu gato é preto'
simple_processor = 'meu gata ah prata'
gan_processor = 'meu fato e puto'
decrappification_processor = 'meu fato é prata'

instance = rouge_metric.RougeMetric(max_n=4, batch_size=2)
instance.store_data(file_reference, reference, simple_processor, gan_processor, decrappification_processor)
instance.compute_metric_value()
instance.export_data('teste.csv')
