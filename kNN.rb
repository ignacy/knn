require './data_set_reader'
require 'linalg'

module Knn
  class Classifier
    include Linalg
    attr_accessor :dataSet, :labels, :k

    def initialize(d, l, k)
      @dataSet = DMatrix.rows(d)
      @labels, @k = l, k
    end

    def classify(example)
      @dataSet = normalize(@dataSet)
      distances = find_distances(example)
      pairs = pair_up(distances)
      votes = gather_votes(votes, pairs)
      freq = votes.inject(Hash.new(0)) { |h,v| h[v] += 1; h }
      votes.sort_by { |v| freq[v] }.last
    end

    def normalize(set)
      mins, maxes = [], []
      set.columns.each_with_index do |column, i|
        maxes[i] = column.to_a.max
        mins[i]  = column.to_a.min
      end

      mins.flatten!
      maxes.flatten!

      new_set = []      
      set.rows.each do |row|
        new_row = []
        row.elems.each_with_index do |cell, idx|
          new_row[idx] = (cell.to_i - mins[idx]) / (maxes[idx] - mins[idx])
        end
        new_set << new_row
      end
      
      DMatrix.rows(new_set)
    end

    def test_classifier_accuracy
      testingSetSize = 0.10
      datingDataMat, datingLabels = DataSetReader.new('datingTestSet.txt').read
      normMat = normalize(datingDataMat)
      m = normMat.size
      numTestVecs = (m * testingSetSize).ceil
      
      errorCount = 0.0
      classIfer = Classifier.new(normMat[0...numTestVecs], datingLabels[0...numTestVecs], 3)
      (0...numTestVecs).each do |i|
        classifierResult = classIfer.classify(normMat[i])
        puts "the classifier came back with: #{classifierResult} and should with #{datingLabels[i]} "
        if (classifierResult != datingLabels[i])
          errorCount += 1.0
        end

        puts "the total error rate is: == #{errorCount/numTestVecs}"
      end
    end


    private

    def find_distances(ex)
      distances = []
      @dataSet.rows.each do |row|
        sum = 0
        row.each_with_index do |el, i|
          sum += ((el - ex[i])**2)
        end
        distances << ::Math.sqrt(sum)
      end
      distances
    end

    def pair_up(distances)
      pairs = []
      distances.each_with_index do |d, i|
        pairs << [d, @labels[i]]
      end
      pairs = pairs.sort_by { |p| p.first }
      return pairs
    end

    def gather_votes(votes, pairs)
      votes = []
      (0...@k).each do |neighbour|
        votes << pairs[0].last
      end
      return votes
    end
  end
end
