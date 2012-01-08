module Knn
  class DataSetReader
    attr_accessor :file
    
    def initialize(f)
      @file = f
    end

    # Read file into data set
    # assume labels are in the last column
    def read
      data = []
      labels = []
      f = File.open(@file, "r") 
      f.each_line do |line|
        elements = line.split(/\s+/)
        data << elements[0...-1]
        labels << elements[-1]
      end

      [data, labels]
    end
  end
end
