package peersim.gossip;

import java.util.Comparator;

public class ArrayIndexComparator implements Comparator<Integer>
{
    private final Double[] arr;

    public ArrayIndexComparator(Double[] array)
    {
        this.arr = array;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++)
        {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
         // Autounbox from Integer to int to use as array indexes
        return arr[index1].compareTo(arr[index2]);
    }
}