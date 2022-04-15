package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {
    POP ("pop", 0D),

    COUNTRY("country", 1D),

    BLUES("blues", 2D),

    JAZZ("jazz", 3D),

    REGGAE("reggae", 4D),

    ROCK("rock", 5D),

    HIPHOP("hip hop", 6D),

    UNKNOWN("Don\'t know :(", -1D);

    private final String name;
    private final Double value;

    Genre(final String name, final Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }

}
