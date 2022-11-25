
CREATE TABLE adresse (
                id_adresse INT NOT NULL,
                code_dep INT NOT NULL,
                commune VARCHAR(30) NOT NULL,
                PRIMARY KEY (id_adresse)
);


CREATE TABLE bien (
                id_bien INT AUTO_INCREMENT NOT NULL,
                nom_voie VARCHAR(30) NOT NULL,
                surface_carrez DECIMAL(5,2) NOT NULL,
                type_local VARCHAR(15) NOT NULL,
                nombre_piece INT NOT NULL,
                adresse INT NOT NULL,
                PRIMARY KEY (id_bien)
);


CREATE TABLE vente (
                id_vente INT AUTO_INCREMENT NOT NULL,
                date DATE NOT NULL,
                val_fonc DECIMAL(10,2) NOT NULL,
                bien INT NOT NULL,
                PRIMARY KEY (id_vente)
);


ALTER TABLE bien ADD CONSTRAINT adresse_bien_fk
FOREIGN KEY (adresse)
REFERENCES adresse (id_adresse)
ON DELETE NO ACTION
ON UPDATE NO ACTION;

ALTER TABLE vente ADD CONSTRAINT bien_vente_fk
FOREIGN KEY (bien)
REFERENCES bien (id_bien)
ON DELETE NO ACTION
ON UPDATE NO ACTION;