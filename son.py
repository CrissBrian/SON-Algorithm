from pyspark import SparkContext
import os
import argparse
import time
from collections import defaultdict, namedtuple

FreqItemset = namedtuple("FreqItemset", ['items', "freq"])

class APriori(object):
	def __init__(self, support: int):
		self._support = support
		self._freq_itemsets = None

	def inference(self, data):
		print("The supprot is:", self._support)
		freq_itemsets = list()
		for i, item_sets in enumerate(self._iteration(data)):
			freq_itemsets.extend([
				FreqItemset(items=items, freq=freq) for items, freq in item_sets.items()
			])
			# print("Iteration:",i, item_sets)
		self._freq_itemsets = freq_itemsets
		return self

	def frequent_itemsets(self):
		return self._freq_itemsets

	def _iteration(self, data):
		# The first pass to count frequent singletons
		item_sets, data = self._first_pass(data=data)
		if len(item_sets) > 0:
			yield item_sets
		else:
			return
		# The second pass to count frequent pairs.
		item_sets, data = self._second_pass(data=data,
							frequent_items=item_sets)
		if len(item_sets) > 0:
			yield item_sets
		else:
			return
		# More passes to count frequent tuple or 4,5....
		set_length = 3 
		while True:
			item_sets, data = self._more_passes(data=data,
									frequent_prev_set=item_sets,
									set_length=set_length)
			if len(item_sets) > 0:
				yield item_sets
			else:
				break
			set_length += 1

	def _first_pass(self, data):
		count_map = defaultdict(int)
		new_data = list()
		for basket in data:
			new_data.append(basket)
			count_map = self._count_singleton(basket=basket, count_map=count_map)
		
		frequent_items = {
			frozenset({key}): val for key, val in count_map.items() if val >= self._support
		}
		return frequent_items, new_data

	def _second_pass(self, data,
					 frequent_items):
		count_map = defaultdict(int)
		new_data = list()
		for basket in data:
			new_data.append(basket)
			count_map = self._count_pair(basket=basket,
										 frequent_items=frequent_items,
										 count_map=count_map)
		frequent_pairs = {
			key: val for key, val in count_map.items() if val >= self._support
		}
		return frequent_pairs, new_data

	def _more_passes(self, data, frequent_prev_set, set_length):
		# find candidates |X+1| from the previous frequent set |X|.
		candidates = self.find_candidates(frequent_set=frequent_prev_set,
										  set_length=set_length)
		count_map = defaultdict(int)
		new_data = list()
		for basket in data:
			new_data.append(basket)
			count_map = self._count_more(basket=basket,
										 candidates=candidates,
										 count_map=count_map)
		frequent_next_set = {
			key: val for key, val in count_map.items() if val >= self._support
		}
		return frequent_next_set, new_data

	@staticmethod
	def _count_singleton(basket, count_map):
		for item in basket:
			count_map[item] += 1
		return count_map

	@staticmethod
	def _count_pair(basket, count_map, frequent_items):
		# filter the basket with frequent_items
		filtered_basket = list({
			item for item in basket
			if frozenset({item}) in frequent_items
		})
		# print('filtered_basket', filtered_basket)
		# if length is less than 2, noting to compute
		if len(filtered_basket) < 2:
			return count_map
		# counting count_map pairs!
		for i, item_x in enumerate(filtered_basket[:-1]):
			for item_y in filtered_basket[i+1:]:
				pair = frozenset({item_x, item_y})
				count_map[pair] += 1
		return count_map

	@staticmethod
	def _count_more(basket, candidates, count_map):
		set_basket = set(basket)
		for candidate in candidates:
			if candidate.issubset(set_basket):
				count_map[candidate] += 1
		return count_map

	@staticmethod
	def find_candidates(frequent_set, set_length):
		# return if all subset is frequent!
		candidates = defaultdict(int)
		for set_x in frequent_set:
			for set_y in frequent_set:
				union_set = set_x | set_y
				if len(union_set) == set_length:
					candidates[union_set] += 1
		return set(
			key for key, val in candidates.items() if val >= set_length*(set_length - 1)
		)

class SON(object):
	def __init__(self, support, num_partition):
		self._support = support
		self._num_partition = num_partition
		self._candidates = 0

	def inference(self, spark_context, data):
		# read data
		rdd = spark_context.parallelize(data)\
							.coalesce(self._num_partition)
		# get candidates from APrior
		self._candidates = rdd.mapPartitions(self._find_candidates)\
						.distinct().collect()
		# reduce all result
		item_sets = rdd.flatMap(
			lambda basket: 
				self._count_candidates(basket, set(self._candidates)))\
			.reduceByKey(lambda x, y: x+y)\
			.filter(lambda candidate: candidate[1] >= self._support)

		frequent_items = list()
		for key, _ in item_sets.collect():
			frequent_items.append(key)
		return frequent_items

	def get_candidates(self):
		return self._candidates

	def _find_candidates(self, iterator):
		# run local APrior algorithm to find candidates!
		real_sup = self._support // self._num_partition
		apriori = APriori(support=real_sup)
		sub_candidates = apriori.inference(iterator)
		candidates = [key for key, _ in sub_candidates.frequent_itemsets()]
		return candidates

	@staticmethod
	def _count_candidates(basket, candidates):
		for c in candidates:
			if c.issubset(set(basket)):
				yield (c, 1)

class TaFeng(object):
	def __init__(self, support: int, spark_context):
		self._support = support
		self._sc = spark_context
		self._rdd = None
		self._freq_itemsets = None
		self._baskets = None

	def readfile(self, input_file_path):
		sc = self._sc
		RDD = sc.textFile(input_file_path)
		### Remove header ###
		rdd_header = RDD.first()
		header = sc.parallelize([rdd_header])
		self._rdd = RDD.subtract(header)

	def data_process(self, threshold):
		rdd = self._rdd.map(self.get_user_set) \
				.reduceByKey(self.get_baskets) \
				.filter(lambda x: len(x[1]) > threshold)
		baskets = rdd.collect()
		self._baskets = [list(i[1]) for i in baskets]

	def run_SON(self, output, num_partition):
		son = SON(support=self._support, num_partition=num_partition)
		self._freq_itemsets = son.inference(spark_context=self._sc, 
										data=self._baskets)
		candidates = son.get_candidates()

		with open(output, 'w') as f:
			pass
		self.print_sets(sets=candidates, title="Candidates", 
						output_file_path=output)
		self.print_sets(sets=self._freq_itemsets, title="Frequent Itemsets",
						output_file_path=output)

	def frequent_itemsets(self):
		return self._freq_itemsets

	@staticmethod
	def get_user_set(x):
		x = x.split(',')
		return [x[0] + x[1], set([int(x[5][1:-1])])]

	@staticmethod
	def get_baskets(a,b):
		return a.union(b)
 
	@staticmethod
	def print_sets(sets, title, output_file_path):
		def toStr(list):
			return [str(i) for i in list]
		def write_list(list):
			if len(list) == 1:
				f.write("('"+list[0]+"')")
			else:
				f.write("('"+list[0]+"'")
				for i in list[1:]:
					f.write(", '"+i+"'")
				f.write(")")
		result = [toStr(sorted(list(s))) for s in sets]
		result = sorted(result, key=lambda x: (len(x), x))
		with open(output_file_path, 'a') as f:
			f.write(title+":\n")
			if len(result) > 0:
				write_list(result[0])
			length = 1
			for lis in result[1:]:
				if len(lis) > length:
					f.write("\n")
					length += 1
				else:
					f.write(",")
				write_list(lis)
			f.write("\n\n")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("threshold", default="5", help="input review file")
	parser.add_argument("support", default="4", help="input review file")
	parser.add_argument("input", default="ta_feng_all_months_merged.csv", help="input review file")
	parser.add_argument("output", default="output.txt", help="output file")
	args = parser.parse_args()

	sc = SparkContext('local[*]', 'FrequentItems')
	sc.setLogLevel("ERROR")
	support = int(args.support)
	threshold = int(args.threshold)

	# start_time = time.time()
	tafeng = TaFeng(support=support, spark_context=sc)
	tafeng.readfile(input_file_path=args.input)
	tafeng.data_process(threshold=threshold)
	### compute the time after we get the Date_User, Product Dataset
	start_time = time.time()
	tafeng.run_SON(output=args.output, num_partition=2)
	print(len(tafeng.frequent_itemsets()))
	total_time = time.time() - start_time
	print("Duration:", total_time)


if __name__ == '__main__':
	main()